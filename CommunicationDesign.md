# **Unity Communication Controller - Complete Specification**

## **CONNECTION DETAILS**

**Server Address**: `127.0.0.1` (localhost)  
**Port**: `65432`  
**Protocol**: ZeroMQ (REQ/REP pattern) with JSON messaging  
**Connection Type**: Client-Server (Unity = REQ Client, Python = REP Server)  
**Receive Timeout**: Auto-configured based on tickrate (minimum 2 seconds, default 3× tick interval)  
**Disconnect Detection**: Server automatically detects client disconnection via timeout and socket status checks

---

## **ZEROMQ COMMUNICATION PATTERN**

### **REQ/REP Pattern Overview**

The server uses ZeroMQ's **Request-Reply (REQ/REP)** pattern:

- **Unity Client (REQ)**: Sends requests (game state) and waits for replies
- **Python Server (REP)**: Receives requests, processes them, and sends replies (steering/config)

### **Important ZeroMQ Rules**

1. **Strict Message Alternation**: Unity MUST send a request before receiving a reply
2. **Synchronous Communication**: Unity blocks until it receives a reply from Python
3. **Connection Format**: `tcp://127.0.0.1:65432`

---

## **INITIAL HANDSHAKE & CONFIGURATION**

### **First Connection Flow**

```

1. Unity: Connect to server
2. Unity: Send first game_state message (handshake)
3. Server: Receive handshake, send configuration reply
4. Unity: Receive config, synchronize tickrate
5. Unity: Send actual game_state messages
6. Server: Process and send steering commands
7. Episodes continue on same connection until:
   - Unity disconnects
   - Server timeout (no message received within receive_timeout)
   - Server shutdown

```

**Important**: Configuration is sent **ONCE per connection**, not per episode. Multiple episodes can run on the same connection without re-handshaking.

### **Configuration Message (Server → Unity)**

**First reply from server contains configuration:**

```json
{
  "type": "config",
  "tickrate": 30,
  "tick_interval_ms": 33.33,
  "max_episode_steps": 1000,
  "message": "Server configuration. Please synchronize your update rate."
}
```

| Field               | Type   | Description                                   |
| ------------------- | ------ | --------------------------------------------- |
| `type`              | string | Always `"config"` for configuration messages  |
| `tickrate`          | int    | Server tickrate in Hz (updates per second)    |
| `tick_interval_ms`  | float  | Time interval between updates in milliseconds |
| `max_episode_steps` | int    | Maximum steps per episode before truncation   |
| `message`           | string | Informational message                         |

---

## **ENVIRONMENT CONFIGURATION**

### **Ray Configuration**

```
Index 0: Forward Ray      - Max Distance: 7.0
Index 1: Forward-Left Ray - Max Distance: 4.5
Index 2: Forward-Right Ray- Max Distance: 4.5
Index 3: Right Ray        - Max Distance: 3.5
Index 4: Left Ray         - Max Distance: 3.5
```

### **Physics Parameters**

- **Max Speed**: `2.5` units
- **Steering Speed Penalty**: `-0.5` units (when steering ≠ 0)
- **Max Episode Steps**: `1000` steps

### **Reward Configuration** (Python-calculated)

- **Survival Reward**: `+0.1` per step
- **Reward Collected**: `+15.0`
- **Collision Penalty**: `-10.0`

---

## **MULTI-EPISODE SESSIONS & DISCONNECTION**

### **Same Connection, Multiple Episodes**

- Client stays connected across episodes
- Server automatically starts new episode when previous ends (if client connected)
- No re-handshake needed between episodes
- Configuration received on first connection applies to all episodes in session

### **Disconnection Detection**

Server detects disconnection through:

1. **Timeout**: No message received within `receive_timeout_ms` (calculated as `max(2000ms, tickrate_interval × 3)`)
2. **Socket Status**: ZMQ socket event checks during episode transitions
3. **Send Failures**: Unable to send response to client

### **Server Behavior on Disconnect**

When client disconnects:

1. Server logs "CLIENT DISCONNECTED" with session statistics
2. Server resets to `WAITING_FOR_CLIENT` state
3. Connection manager state changes to `LISTENING`
4. Server ready to accept new client connection
5. Episode/step counters preserved for statistics

### **Reconnection Protocol**

If Unity needs to reconnect:

1. Close existing socket
2. Create new REQ socket
3. Connect to server
4. Send handshake (first game_state message)
5. Receive new configuration
6. Resume normal operation

---

## **UNITY → PYTHON (Game State Request)**

### **Message Structure**

```json
{
  "message": "game_state",
  "id": 123,
  "gameState": {
    "rayDistances": [7.0, 4.5, 4.5, 3.5, 3.5],
    "rayHits": [0, 0, 0, 0, 0],
    "carSpeed": 2.5,
    "rewardCollected": 0,
    "collisionDetected": 0,
    "respawns": 0,
    "elapsedTime": 10.5
  }
}
```

### **Field Specifications**

| Field               | Type    | Required | Values/Range     | Description                                        |
| ------------------- | ------- | -------- | ---------------- | -------------------------------------------------- |
| `message`           | string  | ✅       | `"game_state"`   | Message type identifier                            |
| `id`                | int     | ✅       | Any positive int | Message sequence number (increment each message)   |
| `rayDistances`      | float[] | ✅       | [0.0, maxDist]   | Distance to nearest obstacle for each ray          |
| `rayHits`           | int[]   | ✅       | 0 or 1           | Ray hit indicator (0=clear, 1=hit)                 |
| `carSpeed`          | float   | ✅       | [0.0, 2.5]       | Current car linear velocity                        |
| `rewardCollected`   | int     | ✅       | 0 or 1           | Signal: 1 if reward collected this frame, else 0   |
| `collisionDetected` | int     | ✅       | 0 or 1           | Signal: 1 if collision occurred this frame, else 0 |
| `respawns`          | int     | ✅       | ≥ 0              | Total number of respawns in episode                |
| `elapsedTime`       | float   | ✅       | ≥ 0.0            | Time elapsed in episode (seconds)                  |

---

## **PYTHON → UNITY (Action & Feedback Reply)**

### **Message Structure**

```json
{
  "steering": 0,
  "reward": 0.1,
  "episode_reward": 15.3,
  "step": 42,
  "total_steps": 1337,
  "episode": 5,
  "total_episodes": 5,
  "terminated": false,
  "truncated": false
}
```

### **Field Specifications**

| Field            | Type  | Description                                          |
| ---------------- | ----- | ---------------------------------------------------- |
| `steering`       | int   | Steering command: -1 (left), 0 (straight), 1 (right) |
| `reward`         | float | Reward received for this step                        |
| `episode_reward` | float | Cumulative reward for current episode                |
| `step`           | int   | Current step number in episode                       |
| `total_steps`    | int   | Total steps across all episodes                      |
| `episode`        | int   | Current episode number                               |
| `total_episodes` | int   | Total episodes completed                             |
| `terminated`     | bool  | Episode ended due to collision/respawn               |
| `truncated`      | bool  | Episode ended due to max steps reached               |

### **Steering Values**

| Value | Direction   | Unity Action               |
| ----- | ----------- | -------------------------- |
| `-1`  | Turn LEFT   | Apply left steering input  |
| `0`   | Go STRAIGHT | No steering input          |
| `1`   | Turn RIGHT  | Apply right steering input |

---

## **COMMUNICATION FLOW**

### **Complete Flow Diagram**

```
Unity                           Python Server
  |                                  |
  |--1. Connect ZeroMQ-------------->| (Listening)
  |                                  |
  |--2. Send handshake (game_state)->| (Receive first message)
  |                                  | (Send configuration)
  |<-3. Receive config --------------|
  |   {tickrate: 30, ...}            |
  | (Synchronize tickrate)           |
  |                                  |
  |--4. Send game_state (id:1) ----->| (Process state, get action)
  |<-5. Receive response ------------|
  |   {steering:0, reward:0.1, ...}  |
  | (Apply steering)                 |
  | (Wait tick_interval)             |
  |                                  |
  |--6. Send game_state (id:2) ----->| (Process state, reward: +15.1)
  |    rewardCollected: 1            | (Log: "Reward collected!")
  |<-7. Receive response ------------|
  |   {steering:1, reward:15.1, ...} |
  | (Apply steering)                 |
  | (Wait tick_interval)             |
  |                                  |
  |--8. Send game_state (id:3) ----->| (Process state, penalty: -9.9)
  |    collisionDetected: 1          | (Log: "Collision detected!")
  |                                  | (Episode ends)
  |<-9. Receive response ------------|
  |   {steering:0, terminated:true}  |
  |                                  | (New episode starts)
  |                                  |
  |--10. Continue loop ------------->|
  |...                               |...
```

---

## **CRITICAL IMPLEMENTATION NOTES**

### **1. Signal Flags (MUST RESET!)**

```csharp
// ❌ WRONG - Flags stay set forever
void OnRewardCollected() {
    rewardCollected = 1;  // Set but never reset
}

// ✅ CORRECT - Reset after sending
int SendGameStateAndGetSteering() {
    // ... send message ...

    // Reset single-frame signals immediately after sending
    currentState.rewardCollected = 0;
    currentState.collisionDetected = 0;

    return steering;
}
```

### **2. Tickrate Synchronization**

```csharp
// ❌ WRONG - Using fixed tickrate
void Update() {
    if (Time.time % 0.033f < Time.deltaTime) {  // Hardcoded 30Hz
        SendGameState();
    }
}

// ✅ CORRECT - Using server's tickrate
void Update() {
    timeSinceLastUpdate += Time.deltaTime;
    if (timeSinceLastUpdate >= tickInterval) {  // Server-provided interval
        SendGameState();
        timeSinceLastUpdate = 0f;
    }
}
```

### **3. Ray Distance Rules**

```csharp
// Return max distance if no hit
float CastRay(Vector3 direction, float maxDistance, int index) {
    if (Physics.Raycast(transform.position, direction, out hit, maxDistance)) {
        rayHits[index] = 1;
        return hit.distance;  // Actual distance
    }
    rayHits[index] = 0;
    return maxDistance;  // ✅ Return max, not 0!
}
```

### **4. Message ID Increment**

```csharp
// ✅ Increment ID for each message
messageId++;  // 1, 2, 3, 4...
```

---

## **EPISODE TERMINATION**

Episodes end when:

1. **Collision**: `collisionDetected = 1`
2. **Respawn**: `respawns > 0`
3. **Truncation**: `step >= 1000`

Server automatically starts new episode if Unity stays connected.

---

## **UNITY DEPENDENCIES**

### **Install NetMQ**

```
Option 1 (Package Manager):
  Assets → Package Manager → Add from git URL
  https://github.com/NetMQ/NetMQ.git

Option 2 (NuGet):
  Install NetMQ package via NuGet for Unity
```

### **Required Using Statements**

```csharp
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
```

---

## **QUICK REFERENCE**

### **Server Configuration (Default)**

- Host: `127.0.0.1`
- Port: `65432`
- Default Tickrate: `30 Hz` (configurable in server.py main())
- Max Episode Steps: `1000`
- Receive Timeout: `max(2000ms, tickrate_interval × 3)`
- Accept Timeout: `1000ms` (for new client connections)

### **Message Types**

- **config**: Server → Unity (first message of connection only, NOT per episode)
- **game_state**: Unity → Server (every tick, all episodes)
- **response**: Server → Unity (steering + rewards + episode stats, every tick)

### **Steering Commands**

- `-1`: Turn Left
- `0`: Go Straight
- `1`: Turn Right

### **Reward Values**

- Survival: `+0.1` per step
- Cube Collection: `+15.0`
- Collision: `-10.0`

```

This concise documentation focuses on what Unity developers need to know:
1. Initial handshake and configuration synchronization
2. Complete, working Unity code example
3. Critical implementation details
4. Communication flow diagram
5. Common pitfalls and solutions
6. Quick reference for key valuesThis concise documentation focuses on what Unity developers need to know:
1. Initial handshake and configuration synchronization
2. Complete, working Unity code example
3. Critical implementation details
4. Communication flow diagram
5. Common pitfalls and solutions
6. Quick reference for key values
```
