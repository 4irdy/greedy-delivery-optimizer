# Assignment 5: Greedy Solution Designer

## Setup Instructions

### 1. Clone this repository and open in VS Code

```bash
git clone <REPO_URL>
cd REPO_NAME_HERE
code .
```

### 2. Generate the test scenarios

Run the data generator to create three delivery optimization scenarios:

```bash
python delivery_generator.py
# You may need to run python3 delivery_generator.py
```

This will create:
- `scenarios/package_prioritization.json` - 50 deliveries with time windows
- `scenarios/truck_loading.json` - 100 packages with weights and priorities
- `scenarios/driver_assignment.json` - 60 deliveries to assign to drivers

### 3. Implement your greedy algorithms

Open `greedy_optimizer.py` and complete the three greedy functions:
1. `maximize_deliveries()` - Activity selection problem
2. `optimize_truck_load()` - Fractional knapsack problem
3. `minimize_drivers()` - Interval scheduling problem

### 4. Test and benchmark

Uncomment the test functions in the `__main__` block:

```python
if __name__ == "__main__":
    test_package_prioritization()    # Verify activity selection works
    test_truck_loading()              # Verify fractional knapsack works
    test_driver_assignment()          # Verify interval scheduling works
    benchmark_scenarios()             # Test on realistic delivery data
```

Then run:

```bash
python greedy_optimizer.py
# or python3 greedy_optimizer.py
```

## The Three Problems

### Problem 1: Package Prioritization (Activity Selection)
**Scenario**: You have 50 deliveries with time windows. You can only do one delivery at a time. Which deliveries should you schedule to maximize the total number completed?

**Greedy strategy**: Choose the delivery that finishes earliest among remaining options.

### Problem 2: Truck Loading (Fractional Knapsack)
**Scenario**: You have 100 packages with different weights and priority values. Your truck can carry 500 lbs. You can deliver partial packages. How do you maximize total priority value delivered?

**Greedy strategy**: Sort packages by priority-to-weight ratio, take packages in order (taking fractions if needed).

### Problem 3: Driver Assignment (Interval Scheduling)
**Scenario**: You have 60 deliveries with time windows. Each driver can handle multiple deliveries if they don't overlap. What's the minimum number of drivers needed?

**Greedy strategy**: Assign each delivery to the first driver who is available, or add a new driver if no one is available.

## What to Submit

Submit two files:
1. **Your completed Python file** (rename to `greedy_optimizer.py`)
2. **Written analysis document** (Google Doc or Word) containing:
   - Greedy Property Justification section (250-300 words)
   - Complexity Analysis section (150-200 words)
   - Total: 400-500 words