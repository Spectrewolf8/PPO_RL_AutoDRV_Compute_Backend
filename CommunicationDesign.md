# **Unity Communication Controller - Complete Specification**

## **CONNECTION DETAILS**

**Server Address**: `127.0.0.1` (localhost)  
**Port**: `65432`  
**Protocol**: TCP Socket with JSON messaging  
**Connection Type**: Client-Server (Unity = Client, Python = Server)

---

## **ENVIRONMENT CONFIGURATION**

### **Ray Configuration**

```
Index 0: Forward Ray     - Max Distance: 7.0
Index 1: Forward-Left Ray - Max Distance: 4.5
Index 2: Forward-Right Ray - Max Distance: 4.5
Index 3: Right Ray       - Max Distance: 3.5
Index 4: Left Ray        - Max Distance: 3.5
```

### **Physics Parameters**

- **Max Speed**: `2.5` units
- **Steering Speed Penalty**: `-0.5` units (when steering ≠ 0)
- **Max Episode Steps**: `1000` steps

### **Reward Configuration** (Python-calculated)

- **Survival Reward**: `+0.1` per step
- **Reward Collected**: `+10.0`
- **Collision Penalty**: `-10.0`

---

## **📤 UNITY → PYTHON (Input Message)**

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

| Field       | Type   | Required | Values/Range                | Description                                      |
| ----------- | ------ | -------- | --------------------------- | ------------------------------------------------ |
| `message`   | string | ✅ Yes   | `"game_state"` or `"reset"` | Message type identifier                          |
| `id`        | int    | ✅ Yes   | Any positive integer        | Message sequence number (increment each message) |
| `gameState` | object | ✅ Yes   | See below                   | Current game state data                          |

### **gameState Object Fields**

| Field               | Type    | Required | Values/Range                          | Description                                        |
| ------------------- | ------- | -------- | ------------------------------------- | -------------------------------------------------- |
| `rayDistances`      | float[] | ✅ Yes   | Array of 5 floats, [0.0, maxDistance] | Distance to nearest obstacle for each ray          |
| `rayHits`           | int[]   | ✅ Yes   | Array of 5 ints, 0 or 1               | Ray hit indicator (0=clear, 1=hit)                 |
| `carSpeed`          | float   | ✅ Yes   | [0.0, 2.5]                            | Current car linear velocity                        |
| `rewardCollected`   | int     | ✅ Yes   | 0 or 1                                | Signal: 1 if reward collected this frame, else 0   |
| `collisionDetected` | int     | ✅ Yes   | 0 or 1                                | Signal: 1 if collision occurred this frame, else 0 |
| `respawns`          | int     | ✅ Yes   | ≥ 0                                   | Total number of respawns in episode                |
| `elapsedTime`       | float   | ✅ Yes   | ≥ 0.0                                 | Time elapsed in episode (seconds)                  |

### **Unity Implementation Requirements**

```csharp
// Pseudo-code for Unity message construction
class GameState {
    public float[] rayDistances = new float[5];  // [Forward, Fwd-Left, Fwd-Right, Right, Left]
    public int[] rayHits = new int[5];           // 0 = clear, 1 = hit
    public float carSpeed;                        // Rigidbody velocity magnitude
    public int rewardCollected;                   // 1 on collection frame, reset to 0 next frame
    public int collisionDetected;                 // 1 on collision frame, reset to 0 next frame
    public int respawns;                          // Total respawns counter
    public float elapsedTime;                     // Time.time or stopwatch
}

// Raycast logic
void UpdateRaycasts() {
    // Ray 0: Forward
    rayDistances[0] = CastRay(transform.forward, 7.0f);

    // Ray 1: Forward-Left (45 degrees)
    rayDistances[1] = CastRay(Quaternion.Euler(0, -45, 0) * transform.forward, 4.5f);

    // Ray 2: Forward-Right (45 degrees)
    rayDistances[2] = CastRay(Quaternion.Euler(0, 45, 0) * transform.forward, 4.5f);

    // Ray 3: Right (90 degrees)
    rayDistances[3] = CastRay(transform.right, 3.5f);

    // Ray 4: Left (-90 degrees)
    rayDistances[4] = CastRay(-transform.right, 3.5f);
}

float CastRay(Vector3 direction, float maxDistance) {
    RaycastHit hit;
    if (Physics.Raycast(transform.position, direction, out hit, maxDistance)) {
        rayHits[index] = 1;
        return hit.distance;
    }
    rayHits[index] = 0;
    return maxDistance;  // Return max if no hit
}

// Speed calculation
void FixedUpdate() {
    float baseSpeed = 2.5f;
    float penalty = 0.5f;

    if (steeringInput != 0) {
        targetSpeed = baseSpeed - penalty;  // 2.0 when steering
    } else {
        targetSpeed = baseSpeed;  // 2.5 when straight
    }

    carSpeed = rigidbody.velocity.magnitude;
}

// Signal management (reset after sending)
void OnRewardCollected() {
    rewardCollected = 1;  // Set flag
}

void OnCollision() {
    collisionDetected = 1;  // Set flag
}

void AfterSendingToServer() {
    rewardCollected = 0;      // Reset flag
    collisionDetected = 0;    // Reset flag
}
```

---

## **📥 PYTHON → UNITY (Output Message)**

### **Message Structure**

```json
{
  "steering": -1
}
```

### **Field Specifications**

| Field      | Type | Required | Values            | Description              |
| ---------- | ---- | -------- | ----------------- | ------------------------ |
| `steering` | int  | ✅ Yes   | `-1`, `0`, or `1` | Steering command for car |

### **Steering Values**

| Value | Direction   | Unity Action               |
| ----- | ----------- | -------------------------- |
| `-1`  | Turn LEFT   | Apply left steering input  |
| `0`   | Go STRAIGHT | No steering input          |
| `1`   | Turn RIGHT  | Apply right steering input |

### **Unity Implementation**

```csharp
// Pseudo-code for Unity response handling
void HandleServerResponse(string jsonResponse) {
    var response = JsonUtility.FromJson<ServerResponse>(jsonResponse);
    int steering = response.steering;

    switch(steering) {
        case -1:
            // Turn left
            ApplySteering(-1f);
            break;
        case 0:
            // Go straight
            ApplySteering(0f);
            break;
        case 1:
            // Turn right
            ApplySteering(1f);
            break;
    }
}

void ApplySteering(float input) {
    // Apply to car controller (Rigidbody, Wheel Colliders, etc.)
    // Example: carController.steerInput = input;
}
```

---

## **🔄 COMMUNICATION FLOW**

### **Episode Lifecycle**

```
1. Unity connects to Python server (127.0.0.1:65432)
2. Python: Episode starts, environment resets
3. Unity: Sends initial game state
4. Loop until episode ends:
   a. Python: Receives game state
   b. Python: Updates environment, calculates reward
   c. Python: Gets action from PPO model
   d. Python: Sends steering command
   e. Unity: Receives steering, applies to car
   f. Unity: Updates physics (1 frame)
   g. Unity: Sends new game state
5. Episode ends when:
   - collisionDetected = 1 (terminated)
   - respawns > 0 (terminated)
   - step count >= 1000 (truncated)
6. Unity: Disconnects or sends reset request
7. Go back to step 1 for new episode
```

### **Timing Diagram**

```
Unity                          Python Server
  |                                |
  |--- Connect TCP Socket -------->|
  |                                | (Episode 1 starts)
  |                                |
  |--- game_state (id:1) --------->|
  |                                | (Process state)
  |                                | (Calculate reward: +0.1)
  |                                | (Get action: 0)
  |<------ steering: 0 ------------|
  | (Apply straight)               |
  | (Update physics)               |
  |                                |
  |--- game_state (id:2) --------->|
  |    rewardCollected: 1          | (Reward: +0.1 + 10.0 = +10.1)
  |                                | (Get action: 1)
  |<------ steering: 1 ------------|
  | (Apply right)                  |
  | (Update physics)               |
  |                                |
  |--- game_state (id:3) --------->|
  |    collisionDetected: 1        | (Reward: +0.1 - 10.0 = -9.9)
  |                                | (Episode terminates)
  |<------ steering: 0 ------------|
  |                                |
  |--- Disconnect ----------------->|
  |                                | (Episode 1 ends)
  |                                |
  |--- Reconnect ------------------>|
  |                                | (Episode 2 starts)
  |...                             |...
```

---

## **IMPORTANT NOTES**

### **Signal Behavior (Critical!)**

- **`rewardCollected`** and **`collisionDetected`** are **single-frame flags**
- Set to `1` on the frame the event occurs
- **MUST be reset to `0` immediately after sending** to server
- Example: If reward collected at frame 100:
  - Frame 100: Send `rewardCollected: 1`
  - Frame 101: Send `rewardCollected: 0`

### **Ray Distance Rules**

- If ray hits nothing: Return the **max distance** for that ray
- If ray hits obstacle: Return the **actual distance** to hit point
- Always clamp distances to their respective max values
- Ray hit indicator is **independent** of distance value

### **Speed Calculation**

- Unity handles all speed/physics calculations
- Python only **reads** the speed value
- Apply steering speed penalty in Unity's physics update
- Send actual measured speed to Python (e.g., `rigidbody.velocity.magnitude`)

### **Episode Management**

- Python automatically terminates on `collisionDetected = 1`
- Unity can request reset by sending `message: "reset"`
- After termination, Unity should disconnect and reconnect for new episode
- Message `id` should increment continuously within an episode

---

## **VALIDATION CHECKLIST**

Before implementing in Unity, verify:

- [ ] TCP socket connects to `127.0.0.1:65432`
- [ ] JSON messages are properly formatted
- [ ] All 7 `gameState` fields are included in every message
- [ ] Ray distances array has exactly 5 float values
- [ ] Ray hits array has exactly 5 int values (0 or 1)
- [ ] Message `id` increments with each message
- [ ] `rewardCollected` and `collisionDetected` reset after sending
- [ ] Speed reflects actual car velocity
- [ ] Steering commands are applied correctly (-1, 0, 1)
- [ ] Episode ends properly on collision
- [ ] Connection handles disconnect/reconnect gracefully

---

## **TEST SCENARIOS**

### **Test 1: Basic Communication**

```json
// Send from Unity
{"message": "game_state", "id": 1, "gameState": {"rayDistances": [7.0,4.5,4.5,3.5,3.5], "rayHits": [0,0,0,0,0], "carSpeed": 0.0, "rewardCollected": 0, "collisionDetected": 0, "respawns": 0, "elapsedTime": 0.0}}

// Expect from Python
{"steering": 0}  // or -1, or 1
```

### **Test 2: Reward Collection**

```json
// Frame where reward is collected
{"message": "game_state", "id": 50, "gameState": {"rayDistances": [5.0,3.0,3.0,2.0,2.0], "rayHits": [0,0,0,0,0], "carSpeed": 2.5, "rewardCollected": 1, "collisionDetected": 0, "respawns": 0, "elapsedTime": 5.0}}

// Next frame (flag reset)
{"message": "game_state", "id": 51, "gameState": {"rayDistances": [4.8,2.9,2.9,1.9,1.9], "rayHits": [0,0,0,0,0], "carSpeed": 2.5, "rewardCollected": 0, "collisionDetected": 0, "respawns": 0, "elapsedTime": 5.1}}
```

### **Test 3: Collision (Episode End)**

```json
// Frame where collision occurs
{
  "message": "game_state",
  "id": 120,
  "gameState": {
    "rayDistances": [0.2, 0.5, 0.5, 0.3, 1.0],
    "rayHits": [1, 1, 1, 1, 0],
    "carSpeed": 1.5,
    "rewardCollected": 0,
    "collisionDetected": 1,
    "respawns": 0,
    "elapsedTime": 12.0
  }
}

// Python will send final steering, then episode terminates
// Unity should disconnect and prepare for new episode
```
