# Live Streaming Specification

## Purpose

This document defines the live streaming protocol between a Quest capture app
and the laptop, the receiver and discovery behavior, and how a streamed session
is recorded to disk.

It owns requirements for:

* wire framing and message schemas
* connection, replacement, and timeout behavior
* zero-configuration endpoint resolution (endpoint file, `adb reverse`, UDP discovery)
* bounded queues and drop policy
* the laptop session recording (the master copy of a streamed session)
* replay of a recorded project as a stream

Recording layouts are specified in `RECORDING_FORMATS.md`. Live reconstruction
behavior is specified in `LIVE_RECONSTRUCTION.md`.

---

# Dependency Boundary

`mq3drecon.streaming` is a lightweight package. It must import and work without
Open3D or any GPU library.

---

# Transport

The headset is a TCP client. The laptop listens on TCP port `9870` by default.
Frames travel over this TCP connection, never through adb.

The framing does not depend on the transport. A future WebSocket transport must
carry exactly one framed message per binary WebSocket message.

---

# Framing

Every message is:

| Field | Size | Meaning |
| --- | --- | --- |
| magic | 4 bytes | ASCII `MQLS` |
| version | u16 | Protocol version, currently `1` |
| type | u16 | Message type id |
| hlen | u32 | Header length in bytes, at most 64 KiB |
| plen | u32 | Payload length in bytes, at most 64 MiB |
| header | `hlen` bytes | UTF-8 JSON object |
| payload | `plen` bytes | Binary payload |

All integers are little-endian.

The receiver must reject the following with `StreamProtocolError` and close the
connection:

* bad magic;
* a different protocol version;
* oversize lengths;
* a non-object header;
* an unknown type id;
* a payload size that does not match the header;
* a stream that ends mid-message.

A clean end of stream between messages ends the connection normally.

JSON numbers may be `Infinity`. For `far_z`, `null` also means infinity.

---

# Messages

| Type id | Name | Header fields | Payload |
| --- | --- | --- | --- |
| 1 | `HELLO` | `device`, `app_version` | none |
| 2 | `SESSION_START` | `session_name`, `resume` (bool), `intrinsics` (`left`/`right` → MRUK intrinsics object, same schema as `*_camera_mruk_intrinsics.json`) | none |
| 3 | `COLOR` | `side`, `frame_index`, `timestamp_us_realtime`, `unix_time_ms`, `width`, `height`, `encoding` (`jpeg` \| `rgba`), `row_order` (`top_down` \| `bottom_up`), `pose` {`position` xyz, `rotation` xyzw}, `pose_ok` | encoded image |
| 4 | `DEPTH` | `side`, `timestamp_ms`, `ovr_timestamp`, `width`, `height`, `encoding` (`f32` \| `f16` \| `u16_ndc`), `create_pose`, `fov` {`left`,`right`,`top`,`down`} tangents, `near_z`, `far_z` | raw NDC depth buffer |
| 5 | `HEARTBEAT` | `time_ms`, `queued`, `dropped` (counts per stream) | none |
| 6 | `SESSION_END` | `reason`, `dropped` | none |
| 7 | `BYE` | `reason` | none |

Field meanings and units match the on-disk recording formats:

* `COLOR` fields mirror MRUK frame metadata, and `DEPTH` fields mirror the depth descriptor CSV.
* Poses are Unity camera-to-world, with quaternions ordered `(x, y, z, w)`.
* Color timestamps are microseconds; depth timestamps are milliseconds.

Payload rules:

* `rgba` payloads are exactly `width * height * 4` bytes.
* `jpeg` payloads are baseline JPEG. `row_order` states the row order of the decoded image.
* Depth payloads are row-major in the same row order as `.raw` depth files.
* `u16_ndc` stores `round(clip(ndc, 0, 1) * 65535)` as little-endian u16. `f16` and `f32` store the NDC value directly.

A connection may carry several sessions. `SESSION_START` with `resume: true`
continues an open session of the same name after a reconnect.

---

# Receiver Behavior

* The receiver accepts one connection at a time. A newer connection replaces an older one: the old connection's disconnect is delivered before the new connection's connect.
* Handler callbacks are serialized.
* A connection with no data for the idle timeout (default 10 s) is closed. Senders must send `HEARTBEAT` at least every 2 s while connected.
* The receiver must never drop frames destined for the session recording. Back-pressure is applied through TCP flow control. The sender drops its oldest queued frames and reports them in `HEARTBEAT` and `SESSION_END`.
* Consumers that may fall behind, such as live reconstruction, must use bounded drop-oldest queues and count what they drop.

---

# Endpoint Resolution

The headset resolves the laptop endpoint in this order:

1. **Endpoint file.** Read `live_stream_endpoint.json` from the app's persistent data directory: `{"host": "<ip>", "port": 9870}`. When wireless adb is connected, the laptop determines its own LAN address as seen from the headset and pushes this file.
2. **`adb reverse`.** Try `127.0.0.1:9870`; the laptop runs `adb reverse tcp:9870 tcp:9870` when adb is connected.
3. **UDP discovery.** Broadcast the probe `MQLS?` to UDP port `9871`. A laptop running the receiver replies by unicast with `{"service": "mq3drecon-live", "protocol_version": 1, "port": <tcp>, "host": <name>}`. Replies with another service name or protocol version must be ignored.

adb failures must never prevent the receiver from starting.

---

# Session Recording

The laptop writes each session to a new directory in the MRUK layout
(`RECORDING_FORMATS.md`). It must refuse to write into an existing non-empty
directory.

* `session_info.json` is `{"sessionFormatVersion": 2, "captureBackend": "MRUK"}`.
* Intrinsics JSON files are written verbatim from `SESSION_START`.
* **Color:**
  * `jpeg` frames are stored as received: `<timestamp_us>.jpg`, top-down. A `bottom_up` JPEG is decoded, flipped, and re-encoded at quality 95.
  * `rgba` frames are stored as bottom-up `<timestamp_us>.rgba`.
  * Metadata rows use the headset app's column set, with `format` `JPEG` or `RGBA32`.
* **Depth:** frames are stored for the configured eyes only (default: left). The default format is `<timestamp_ms>.png` (16-bit NDC); `raw` gives float32 `<timestamp_ms>.raw` (`RecordingOptions`). Descriptor rows are identical to the headset's.
* **Duplicates:** a repeated frame (same kind, side, and timestamp) is ignored and counted.
* **Finalizing:**
  * writes `mruk_stereo_pairs.csv` by greedy one-to-one nearest-timestamp pairing (smallest difference first, rows ordered by left timestamp);
  * writes `live_session.json` with frame counts, duplicates, pair count, end reason, and headset drop counts.

---

# Replay

A recorded project can be replayed as a stream. Replay emits:

* `SESSION_START`;
* all valid color and depth frames in capture-time order;
* `SESSION_END` (`replay_complete`).

`.rgba` frames are re-encoded as top-down JPEG. Replay at `speed` 1 preserves
capture timing, and `speed <= 0` sends as fast as possible.
