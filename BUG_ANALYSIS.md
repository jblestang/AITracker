# Bug Analysis: 3 Aircraft Tracking Issues

## Critical Bugs Identified

### 1. **CRITICAL: Conflict Resolution Only Runs for 2+ Tracks AND 2+ Measurements**
**Location:** `src/tracker.rs:266`
**Issue:** Conflict detection only runs when `tracks.len() >= 2 && measurements.len() >= 2`. This means:
- With 3 tracks and 1 measurement, conflicts won't be detected
- All 3 tracks might independently associate with the same measurement
- No conflict resolution occurs, leading to incorrect associations

**Impact:** High - Causes incorrect track-measurement associations when measurement count is low

### 2. **CRITICAL: Shared `prev_track_position` Across All Tracks**
**Location:** `src/tracker.rs:632`
**Issue:** `self.prev_track_position` is a single shared value that gets overwritten for each track in the update loop. This means:
- Only the last track's position is stored
- Velocity estimation for earlier tracks uses wrong previous position
- Breaks velocity estimation for all tracks except the last one

**Impact:** High - Breaks velocity estimation for multiple tracks

### 3. **BUG: Association Probabilities Computed Twice**
**Location:** `src/tracker.rs:270-277` and `424-435`
**Issue:** Association probabilities are computed once for conflict detection, then recomputed in the update loop. This:
- Is inefficient (double computation)
- Could lead to inconsistencies if track state changes
- The resolved associations might not match the recomputed ones

**Impact:** Medium - Performance and potential correctness issues

### 4. **BUG: Conflict Resolution Doesn't Ensure Fair Assignment**
**Location:** `src/tracker.rs:309-403`
**Issue:** When resolving conflicts:
- Tracks that already have good assignments are penalized (0.7x score)
- But tracks with NO assignments are not prioritized
- A track might get multiple measurements while others get none
- No mechanism to ensure all tracks get at least one measurement

**Impact:** Medium - Unfair assignment distribution

### 5. **BUG: M/N Confirmation Check Happens After Association**
**Location:** `src/tracker.rs:670` and `814-816`
**Issue:** 
- Tracks are confirmed AFTER the update (line 670)
- But unassociated measurement check uses confirmed tracks (line 815)
- This means a track might be used for association before it's confirmed
- Could lead to false track initiation from clutter

**Impact:** Medium - Could cause false track creation

### 6. **BUG: Conflict Resolution Normalization Per Track**
**Location:** `src/tracker.rs:378-397`
**Issue:** When probabilities are reduced and normalized per track:
- Each track's probabilities are normalized independently
- Global assignment optimality is not guaranteed
- A measurement might have high probability for multiple tracks after normalization
- No global consistency check

**Impact:** Medium - Suboptimal global assignments

### 7. **INCONSISTENCY: Velocity Estimation Uses Wrong Previous Position**
**Location:** `src/tracker.rs:572-609`
**Issue:** Velocity estimation uses `self.prev_track_position` which is shared across tracks:
- Uses position from last updated track, not the current track
- Breaks velocity estimation for all tracks except the last one
- Should use track-specific previous position

**Impact:** High - Incorrect velocity estimates

### 8. **BUG: Initial Track Selection Doesn't Filter Clutter**
**Location:** `src/tracker.rs:203-234`
**Issue:** Initial track selection uses distance-based clustering but:
- Doesn't consider which measurements are most likely true targets
- Could initialize tracks from clutter if clutter is far enough apart
- No quality check on initial measurements

**Impact:** Low-Medium - Could initialize false tracks

## Proposed Fixes

### Fix 1: Remove Measurement Count Requirement for Conflict Detection
- Change condition from `tracks.len() >= 2 && measurements.len() >= 2` to `tracks.len() >= 2`
- This ensures conflicts are detected even with 1 measurement

### Fix 2: Store Previous Position Per Track
- Add `prev_position: Option<(Vector3<f64>, f64)>` to `Track` struct
- Update it in the track update loop
- Use track-specific previous position for velocity estimation

### Fix 3: Reuse Pre-computed Association Probabilities
- Store the initial associations and reuse them in the update loop
- Only recompute if track state has changed significantly

### Fix 4: Prioritize Tracks Without Assignments
- In conflict resolution, give bonus to tracks that have no good assignments
- Ensure each track gets at least one measurement when possible

### Fix 5: Check Confirmation Before Association
- Move confirmation check to before association probability computation
- Or use a separate flag for "can be used for association"

### Fix 6: Global Assignment Optimization
- After conflict resolution, verify global consistency
- Ensure no measurement is assigned to multiple tracks with high probability

### Fix 7: Track-Specific Previous Position
- Store previous position in Track struct
- Use it for velocity estimation instead of shared value

### Fix 8: Quality-Based Initial Track Selection
- Use association probability or likelihood to select initial measurements
- Prefer measurements that are more likely to be true targets

