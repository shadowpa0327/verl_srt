# gRPC IPv6 Address Resolution Issue

## Status

**Fixed** - Updated `_get_server_addresses()` in Python to use gRPC URI scheme prefixes.

This approach works for both IPv4 and IPv6, and is robust in Ray actor environments
where shell environment variables may not propagate.

## Observed Error

```
Failed to update cache on server [[2605:340:cdb1:6400::a7c:2fac]]:6378 for request 377: DNS resolution failed
(SRTDAPOTaskRunner pid=39990, ip=[2605:340:cdb1:6400::a7c:172]) Failed to update cache on server [[2605:340:cdb1:6400::a7c:172]]:6378 for request 377: DNS resolution failed
```

## Root Cause

The issue is in how gRPC's `CreateChannel()` handles IPv6 addresses vs how `AddListeningPort()` handles them.

### Client Side (suffix_cache_updater.cc)

```cpp
// Line 62: Creates address in [ipv6]:port format
addresses.push_back("[" + ip + "]:6378");

// Line 76: Passes to CreateChannel
auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
```

When you pass `[2605:340:cdb1:6400::a7c:2fac]:6378` to `CreateChannel()`, gRPC's default name resolver (`dns:`) treats the entire string `[2605:340:cdb1:6400::a7c:2fac]` as a **hostname** and attempts DNS resolution, which fails.

### Server Side (rollout_cache_server.cc)

```cpp
// Line 97: Server uses [::]:port format
server_address_(server_address.empty() ? "[::]:6378" : server_address)

// Line 157: Passes to AddListeningPort
builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
```

This works because `AddListeningPort()` correctly parses `[ipv6]:port` format for binding.

## gRPC API Asymmetry

| Function | IPv6 Format | Works? | Reason |
|----------|-------------|--------|--------|
| `AddListeningPort("[::]:6378", ...)` | `[ipv6]:port` | Yes | Direct socket binding |
| `CreateChannel("[ipv6]:6378", ...)` | `[ipv6]:port` | No | Uses DNS resolver |

## How gRPC Name Resolution Works

When calling `CreateChannel(address, ...)`, gRPC uses a URI-based name resolution system:

| You Pass | gRPC Interprets As | Result |
|----------|-------------------|--------|
| `localhost:6378` | `dns:///localhost:6378` | DNS lookup for "localhost" - works |
| `192.168.1.1:6378` | `dns:///192.168.1.1:6378` | DNS resolver recognizes IPv4 literal - works |
| `[2605:340::a7c]:6378` | `dns:///[2605:340::a7c]:6378` | DNS lookup for "[2605:340::a7c]" - fails |
| `ipv6:[2605:340::a7c]:6378` | Direct IPv6 connection | Works |

The DNS resolver happens to recognize IPv4 dotted-decimal format as a literal, but does **not** recognize IPv6 bracket notation.

## Affected Files

Both implementations have the same bug:

- `recipe/srt/srt_plugin/shm_cache/cache_updater/suffix_cache_updater.cc:62`
- `recipe/specRL/spec_rl_cache_impl/specrl/cache_updater/suffix_cache_updater.cc:62`

## Fix (Applied)

Updated `shared_memory_cache_manager.py` with robust IP normalization using Python's
`ipaddress` module:

```python
def _normalize_ip(self, ip: str) -> tuple[str, bool]:
    """Normalize IP address and determine if it's IPv6."""
    import ipaddress

    # Strip brackets and whitespace
    ip = ip.strip().strip('[]')

    try:
        addr = ipaddress.ip_address(ip)
        return str(addr), isinstance(addr, ipaddress.IPv6Address)
    except ValueError:
        # Fallback to heuristic
        logger.warning(f"Could not parse IP address '{ip}', using heuristic")
        return ip, ':' in ip

def _get_server_addresses(self) -> List[str]:
    addresses = []
    for s in self._cache_servers:
        ip, is_ipv6 = self._normalize_ip(s['ip'])
        port = s['port']
        if is_ipv6:
            addresses.append(f"ipv6:[{ip}]:{port}")
        else:
            addresses.append(f"ipv4:{ip}:{port}")
    return addresses
```

**Handles edge cases:**
- Bracketed IPv6: `[2605:340::1]` → `ipv6:[2605:340::1]:port`
- Raw IPv6: `2605:340::1` → `ipv6:[2605:340::1]:port`
- IPv4: `192.168.1.100` → `ipv4:192.168.1.100:port`
- Whitespace: `  [2605:340::1]  ` → normalized
- Compressed IPv6: `::1`, `fe80::1` → correctly detected

**Previous bug:** Double brackets `ipv6:[[addr]]:port` caused:
```
Failed gpr_split_host_port([[addr]]:port, ...)
the target uri is not valid
```

This fix is applied at the Python level where addresses are formatted before passing
to the C++ `SuffixCacheUpdater`. This is more robust than environment variables because:
1. Works in Ray actor environments where shell env vars may not propagate
2. Explicit about address type (no reliance on resolver behavior)
3. No C++ code changes needed

## Alternative Fix (Environment Variable)

If you cannot modify Python code, set this environment variable:

```bash
export GRPC_DNS_RESOLVER=native
```

**Note**: This may not work reliably in Ray actors since environment variables from
the shell may not propagate to Ray worker processes.

## c-ares Version Dependency

The DNS resolution failure depends on the c-ares version:

| c-ares Version | Behavior with `[ipv6]:port` |
|----------------|----------------------------|
| 1.18.1+ | Recognizes bracketed IPv6 as literal (works) |
| Older versions | Treats `[ipv6]` as hostname, DNS lookup fails |

This machine has c-ares **1.18.1** which handles bracketed IPv6 correctly.
Your cluster likely has an older version that fails.

**The fix (`ipv6:` prefix) works on ALL versions** because it bypasses
DNS resolution entirely.

## Verification

The fix was verified against official gRPC documentation:

**Official format**: `ipv6:[address]:port` (single colon after scheme, brackets around address)

**Official examples**:
- `ipv6:[2607:f8b0:400e:c00::ef]:443`
- `ipv6:[::]:1234`

**Our fix produces**: `ipv6:[2605:340:cdb1:6400::a7c:2fac]:6378` - matches the official format.

## References

- [gRPC Name Resolution (C++)](https://grpc.github.io/grpc/cpp/md_doc_naming.html)
- [gRPC Naming Documentation (GitHub)](https://github.com/grpc/grpc/blob/master/doc/naming.md)
- [gRPC DNS Resolution Issues](https://github.com/grpc/grpc/issues/21308)
- gRPC URI Schemes: `dns:`, `ipv4:`, `ipv6:`, `unix:`, `passthrough:`
