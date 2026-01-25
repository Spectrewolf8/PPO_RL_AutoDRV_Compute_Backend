# **Unity Communication Controller - Complete Specification**

## **CONNECTION DETAILS**

**Server Address**: `127.0.0.1` (localhost)  
**Port**: `65432`  
**Protocol**: ZeroMQ (REQ/REP pattern) with JSON messaging  
**Connection Type**: Client-Server (Unity = REQ Client, Python = REP Server)

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

```

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

## **📤 UNITY → PYTHON (Game State Request)**

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

## **📥 PYTHON → UNITY (Action & Feedback Reply)**

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

## **🔧 UNITY IMPLEMENTATION**

### **Complete Unity Client Example**

```csharp
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class PythonServerClient : MonoBehaviour
{
    private RequestSocket client;
    private string serverAddress = "tcp://127.0.0.1:65432";

    // Tickrate synchronization
    private float tickInterval = 0.033f;  // Default 30Hz, will be updated from server
    private float timeSinceLastUpdate = 0f;

    // Message tracking
    private int messageId = 0;
    private bool isConfigured = false;

    // Game state
    private GameState currentState;
    private int currentSteering = 0;

    void Start()
    {
        InitializeConnection();
    }

    void InitializeConnection()
    {
        try
        {
            // Initialize NetMQ
            AsyncIO.ForceDotNet.Force();
            client = new RequestSocket();
            client.Connect(serverAddress);
            Debug.Log($"Connected to Python server at {serverAddress}");

            // Send handshake and receive configuration
            SendHandshakeAndConfigureSelf();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to server: {e.Message}");
        }
    }

    void SendHandshakeAndConfigureSelf()
    {
        // Send first message as handshake
        GameStateMessage handshake = new GameStateMessage
        {
            message = "game_state",
            id = messageId++,
            gameState = InitializeGameState()
        };

        string json = JsonUtility.ToJson(handshake);
        client.SendFrame(json);

        // Receive configuration from server
        string configJson = client.ReceiveFrameString();
        ServerConfig config = JsonUtility.FromJson<ServerConfig>(configJson);

        if (config.type == "config")
        {
            // Synchronize with server tickrate
            tickInterval = config.tick_interval_ms / 1000f;  // Convert to seconds
            isConfigured = true;

            Debug.Log($"✓ Server Configuration Received:");
            Debug.Log($"  Tickrate: {config.tickrate} Hz");
            Debug.Log($"  Interval: {tickInterval}s ({config.tick_interval_ms}ms)");
            Debug.Log($"  Max Episode Steps: {config.max_episode_steps}");
        }
        else
        {
            Debug.LogError("Expected config message but received something else!");
        }
    }

    void Update()
    {
        if (!isConfigured) return;

        timeSinceLastUpdate += Time.deltaTime;

        // Send updates at server's tickrate
        if (timeSinceLastUpdate >= tickInterval)
        {
            UpdateGameState();
            ServerResponse response = SendGameStateAndGetResponse();

            // Apply steering
            ApplySteering(response.steering);

            // Update UI with feedback
            UpdateUI(response);

            // Handle episode end
            if (response.terminated || response.truncated)
            {
                OnEpisodeEnd(response);
            }

            timeSinceLastUpdate = 0f;
        }
    }

    void UpdateGameState()
    {
        // Update raycasts
        UpdateRaycasts();

        // Update car speed
        currentState.carSpeed = GetComponent<Rigidbody>().velocity.magnitude;

        // Update elapsed time
        currentState.elapsedTime += Time.deltaTime;
    }

    ServerResponse SendGameStateAndGetResponse()
    {
        GameStateMessage msg = new GameStateMessage
        {
            message = "game_state",
            id = messageId++,
            gameState = currentState
        };

        string json = JsonUtility.ToJson(msg);
        client.SendFrame(json);

        string response = client.ReceiveFrameString();
        ServerResponse resp = JsonUtility.FromJson<ServerResponse>(response);

        // Reset single-frame signals after sending
        currentState.rewardCollected = 0;
        currentState.collisionDetected = 0;

        return resp;
    }

    void UpdateRaycasts()
    {
        // Ray 0: Forward (7.0 max)
        currentState.rayDistances[0] = CastRay(transform.forward, 7.0f, 0);

        // Ray 1: Forward-Left 45° (4.5 max)
        currentState.rayDistances[1] = CastRay(
            Quaternion.Euler(0, -45, 0) * transform.forward, 4.5f, 1);

        // Ray 2: Forward-Right 45° (4.5 max)
        currentState.rayDistances[2] = CastRay(
            Quaternion.Euler(0, 45, 0) * transform.forward, 4.5f, 2);

        // Ray 3: Right 90° (3.5 max)
        currentState.rayDistances[3] = CastRay(transform.right, 3.5f, 3);

        // Ray 4: Left -90° (3.5 max)
        currentState.rayDistances[4] = CastRay(-transform.right, 3.5f, 4);
    }

    float CastRay(Vector3 direction, float maxDistance, int index)
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, direction, out hit, maxDistance))
        {
            currentState.rayHits[index] = 1;
            return hit.distance;
        }
        currentState.rayHits[index] = 0;
        return maxDistance;
    }

    void ApplySteering(int steering)
    {
        currentSteering = steering;
        float steerInput = steering;  // -1, 0, or 1

        // Apply to your car controller
        // Example: GetComponent<CarController>().SetSteering(steerInput);
    }

    void UpdateUI(ServerResponse response)
    {
        // Update UI text elements (assign these in Inspector)
        // rewardText.text = $"Reward: {response.reward:F2}";
        // episodeRewardText.text = $"Episode Total: {response.episode_reward:F2}";
        // stepText.text = $"Step: {response.step} / 1000";
        // episodeText.text = $"Episode: {response.episode}";
        // totalStepsText.text = $"Total Steps: {response.total_steps}";
    }

    void OnEpisodeEnd(ServerResponse response)
    {
        string reason = response.terminated ? "Collision/Respawn" : "Max Steps";
        Debug.Log($"Episode {response.episode} ended ({reason}). Total Reward: {response.episode_reward:F2}");

        // Optional: Show episode summary UI
        // episodeSummaryPanel.SetActive(true);
        // summaryText.text = $"Episode {response.episode}\nReward: {response.episode_reward:F2}\nSteps: {response.step}";
    }

    // Signal methods (call these when events occur)
    public void OnRewardCollected()
    {
        currentState.rewardCollected = 1;
    }

    public void OnCollisionDetected()
    {
        currentState.collisionDetected = 1;
    }

    void OnDestroy()
    {
        client?.Close();
        NetMQConfig.Cleanup();
    }

    // Data classes
    [System.Serializable]
    public class GameStateMessage
    {
        public string message;
        public int id;
        public GameState gameState;
    }

    [System.Serializable]
    public class GameState
    {
        public float[] rayDistances = new float[5];
        public int[] rayHits = new int[5];
        public float carSpeed;
        public int rewardCollected;
        public int collisionDetected;
        public int respawns;
        public float elapsedTime;
    }

    [System.Serializable]
    public class ServerConfig
    {
        public string type;
        public int tickrate;
        public float tick_interval_ms;
        public int max_episode_steps;
        public string message;
    }

    [System.Serializable]
    public class ServerResponse
    {
        public int steering;
        public float reward;
        public float episode_reward;
        public int step;
        public int total_steps;
        public int episode;
        public int total_episodes;
        public bool terminated;
        public bool truncated;
    }

    GameState InitializeGameState()
    {
        currentState = new GameState();
        return currentState;
    }
}
```

---

## **🔄 COMMUNICATION FLOW**

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

## **⚠️ CRITICAL IMPLEMENTATION NOTES**

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

## **📊 EPISODE TERMINATION**

Episodes end when:

1. **Collision**: `collisionDetected = 1`
2. **Respawn**: `respawns > 0`
3. **Truncation**: `step >= 1000`

Server automatically starts new episode if Unity stays connected.

---

## **🛠️ UNITY DEPENDENCIES**

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

## **🐛 TROUBLESHOOTING**

| Problem                  | Solution                                      |
| ------------------------ | --------------------------------------------- |
| "Address already in use" | Restart Python server                         |
| Unity hangs on send      | Check server is running, verify IP/port       |
| Tickrate mismatch        | Ensure config handshake completes             |
| Steering not applied     | Check steering values are exactly -1, 0, or 1 |
| Rewards not detected     | Verify signal flags reset after each send     |
| Connection drops         | Implement try-catch and reconnection logic    |

---

## **📋 QUICK REFERENCE**

### **Server Configuration (Default)**

- Host: `127.0.0.1`
- Port: `65432`
- Default Tickrate: `30 Hz`
- Max Episode Steps: `1000`

### **Message Types**

- **config**: Server → Unity (first message only)
- **game_state**: Unity → Server (every tick)
- **response**: Server → Unity (steering + rewards + episode stats)

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
