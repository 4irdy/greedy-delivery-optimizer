"""
Data Generator for Greedy Algorithms Assignment
Generates three delivery optimization scenarios for testing greedy algorithms.
"""

import random
import json
import os

def generate_scenarios():
    """Generate three delivery optimization scenarios."""
    
    # Create scenarios directory
    if not os.path.exists("scenarios"):
        os.makedirs("scenarios")
    
    print("Generating delivery scenarios...\n")
    
    # ========================================================================
    # Scenario 1: Package Prioritization (Activity Selection)
    # ========================================================================
    print("Scenario 1: Package Prioritization")
    print("  Problem: Schedule maximum deliveries within time windows")
    print("  Size: 50 deliveries with overlapping time windows")
    
    # Generate time windows (start, end) for deliveries
    time_windows = []
    for i in range(50):
        start = random.randint(8, 16)  # Start between 8am and 4pm
        duration = random.randint(1, 4)  # Delivery takes 1-4 hours
        end = start + duration
        time_windows.append({
            "delivery_id": f"PKG-{i+1:03d}",
            "start": start,
            "end": end
        })
    
    with open("scenarios/package_prioritization.json", "w") as f:
        json.dump(time_windows, f, indent=2)
    
    print("  ✓ Generated: scenarios/package_prioritization.json\n")
    
    # ========================================================================
    # Scenario 2: Truck Loading (Fractional Knapsack)
    # ========================================================================
    print("Scenario 2: Truck Loading Optimization")
    print("  Problem: Maximize package priority within weight limit")
    print("  Size: 100 packages, truck capacity 500 lbs")
    
    # Generate packages with weight and priority value
    packages = []
    for i in range(100):
        weight = random.randint(5, 50)  # 5-50 lbs
        priority = random.randint(10, 200)  # Priority value 10-200
        packages.append({
            "package_id": f"PKG-{i+1:03d}",
            "weight": weight,
            "priority": priority,
            "ratio": round(priority / weight, 2)
        })
    
    scenario_data = {
        "packages": packages,
        "truck_capacity": 500
    }
    
    with open("scenarios/truck_loading.json", "w") as f:
        json.dump(scenario_data, f, indent=2)
    
    print("  ✓ Generated: scenarios/truck_loading.json\n")
    
    # ========================================================================
    # Scenario 3: Driver Assignment (Interval Scheduling)
    # ========================================================================
    print("Scenario 3: Driver Assignment")
    print("  Problem: Assign deliveries to minimum number of drivers")
    print("  Size: 60 deliveries with various time windows")
    
    # Generate delivery time intervals
    deliveries = []
    for i in range(60):
        start = random.randint(8, 18)  # Start between 8am and 6pm
        duration = random.randint(1, 3)  # Delivery takes 1-3 hours
        end = start + duration
        deliveries.append({
            "delivery_id": f"DEL-{i+1:03d}",
            "start": start,
            "end": end
        })
    
    with open("scenarios/driver_assignment.json", "w") as f:
        json.dump(deliveries, f, indent=2)
    
    print("  ✓ Generated: scenarios/driver_assignment.json\n")
    
    print("="*70)
    print("Scenario generation complete!")
    print("\nYou can now implement your greedy algorithms in greedy_optimizer.py")
    print("and use these scenarios to test performance.\n")

if __name__ == "__main__":
    generate_scenarios()