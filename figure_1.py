import numpy as np
import matplotlib.pyplot as plt


def minsnappolytraj(
    waypoints, timepoints, n_samples, min_segment_time=4, max_segment_time=60
):
    """
    Generate minimum snap polynomial trajectories
    This is a simplified version since the MATLAB minsnappolytraj is quite complex
    """
    # Get dimensions
    dim = waypoints.shape[0]  # number of dimensions (usually 3 for x,y,z)
    n_waypoints = waypoints.shape[1]  # number of waypoints
    n_segments = n_waypoints - 1  # number of trajectory segments

    # Generate time samples
    t_samples = np.linspace(timepoints[0], timepoints[-1], n_samples)

    # Initialize arrays for position, velocity, acceleration, jerk, snap
    pos = np.zeros((dim, n_samples))
    vel = np.zeros((dim, n_samples))
    acc = np.zeros((dim, n_samples))
    jerk = np.zeros((dim, n_samples))
    snap = np.zeros((dim, n_samples))

    # For each dimension, create a cubic spline interpolation
    for d in range(dim):
        # Create cubic spline
        cs = np.polynomial.polynomial.Polynomial.fit(timepoints, waypoints[d, :], deg=5)

        # Evaluate at sample points
        for i, t in enumerate(t_samples):
            pos[d, i] = cs(t)
            # Derivatives (approximated)
            if i > 0:
                dt = t_samples[i] - t_samples[i - 1]
                vel[d, i] = (pos[d, i] - pos[d, i - 1]) / dt
                if i > 1:
                    acc[d, i] = (vel[d, i] - vel[d, i - 1]) / dt
                    if i > 2:
                        jerk[d, i] = (acc[d, i] - acc[d, i - 1]) / dt
                        if i > 3:
                            snap[d, i] = (jerk[d, i] - jerk[d, i - 1]) / dt

    # Create a dummy PPoly object (not fully functional as MATLAB's)
    pp = None

    return pos, vel, acc, jerk, snap, pp, timepoints, t_samples


# Clear previous plots
plt.close("all")

# Trajectory 1
startPose = np.array([6, 35, 25, 0.7, 0.2, 0, 0.1])  # [x y z qw qx qy qz]
goalPose = np.array([150, 180, 35, 0.3, 0, 0.1, 0.6])

waypoints = np.array(
    [
        [6.0000, 35.0000, 25.0000, 0.7000, 0.2000, 0, 0.1000],
        [50.7795, 55.2933, 15.8930, 0.9832, -0.0499, -0.0487, 0.1688],
        [36.4962, 77.7424, 41.8276, 0.6369, 0.4115, 0.6508, 0.0391],
        [73.6065, 111.1757, 39.5953, 0.6649, 0.3148, 0.5571, 0.3852],
        [72.4807, 156.0036, 17.4798, 0.4673, 0.1736, 0.3921, 0.7732],
        [109.1714, 188.9437, 9.2032, 0.3519, 0.0896, 0.8990, 0.2448],
        [150.0000, 180.0000, 35.0000, 0.3000, 0, 0.1000, 0.6000],
    ]
)

n_waypoints = 7

# Calculate the distance between waypoints
distance = np.zeros(n_waypoints)
for i in range(1, n_waypoints):
    distance[i] = np.linalg.norm(waypoints[i, 0:3] - waypoints[i - 1, 0:3])

# Assume a UAV speed of 1 m/s and calculate time taken to reach each waypoint
UAVspeed = 1
timepoints = np.cumsum(distance / UAVspeed)
timepoints[0] = 0  # Make sure the first timepoint is 0
n_samples = 70

# Compute states along the trajectory
initialStates, qd, qdd, qddd, qdddd, pp, tPoints, tSamples = minsnappolytraj(
    waypoints.T[0:3, :], timepoints, n_samples, min_segment_time=12, max_segment_time=60
)

# Trajectory 2
startPose2 = np.array([50, -5, 20, 0.2, -0.5, 0, 0.1])  # [x y z qw qx qy qz]
goalPose2 = np.array([185, 150, 30, -0.5, 0, 0.1, 0.6])

waypoints2 = np.array(
    [
        [50.0000, -5.0000, 20.0000, 0.2000, -0.5000, 0, 0.1000],
        [97.7303, 9.7600, 21.9631, 0.2207, -0.9217, 0.1513, -0.2809],
        [140.0131, 20.3124, 46.4721, 0.0666, -0.8445, -0.3611, -0.3898],
        [156.2825, 67.2138, 40.5150, 0.3330, -0.6240, -0.3236, -0.6285],
        [172.5520, 114.1152, 34.5579, 0.5401, -0.2924, -0.2285, -0.7554],
        [185.0000, 150.0000, 30.0000, -0.5000, 0, 0.1000, 0.6000],
    ]
)

n_waypoints2 = 6

distance2 = np.zeros(n_waypoints2)
for i in range(1, n_waypoints2):
    distance2[i] = np.linalg.norm(waypoints2[i, 0:3] - waypoints2[i - 1, 0:3])

# Assume a UAV speed and calculate time taken to reach each waypoint
UAVspeed2 = 0.832265191275853
timepoints2 = np.cumsum(distance2 / UAVspeed2)
timepoints2[0] = 0  # Make sure the first timepoint is 0
n_samples2 = n_samples

# Compute states along the trajectory
initialStates2, qd2, qdd2, qddd2, qdddd2, pp2, tPoints2, tSamples2 = minsnappolytraj(
    waypoints2.T[0:3, :],
    timepoints2,
    n_samples2,
    min_segment_time=4,
    max_segment_time=60,
)

# Trajectory 3
startPose3 = np.array([6, 160, 45, 0.7, 0.2, 0, 0.1])  # [x y z qw qx qy qz]
goalPose3 = np.array([150, 35, 55, 0.3, 0, 0.1, 0.6])

waypoints3 = np.array(
    [
        [6.0000, 160.0000, 45.0000, 0.7000, 0.2000, 0, 0.1000],
        [43.7065, 127.2687, 47.6185, 0.8976, 0.2176, 0.0453, 0.3808],
        [91.5085, 141.5558, 44.3353, 0.8594, -0.1280, -0.1044, 0.4838],
        [115.4762, 97.8931, 48.7053, 0.7246, -0.0792, -0.0004, 0.6846],
        [136.0496, 95.2824, 3.2111, 0.7628, -0.5892, -0.1141, 0.2407],
        [140.3821, 50.0147, 23.9869, 0.3113, -0.0747, 0.0026, 0.9474],
        [150.0000, 35.0000, 55.0000, 0.3000, 0, 0.1000, 0.6000],
    ]
)

n_waypoints3 = 7
distance3 = np.zeros(n_waypoints3)
for i in range(1, n_waypoints3):
    distance3[i] = np.linalg.norm(waypoints3[i, 0:3] - waypoints3[i - 1, 0:3])

# Assume a UAV speed and calculate time taken to reach each waypoint
UAVspeed3 = 0.998246044231461
timepoints3 = np.cumsum(distance3 / UAVspeed3)
timepoints3[0] = 0  # Make sure the first timepoint is 0
n_samples3 = n_samples

# Compute states along the trajectory
initialStates3, qd3, qdd3, qddd3, qdddd3, pp3, tPoints3, tSamples3 = minsnappolytraj(
    waypoints3.T[0:3, :],
    timepoints3,
    n_samples3,
    min_segment_time=4,
    max_segment_time=60,
)

# Trajectory 4
startPose4 = np.array([0, 100, 45, 0.7, 0.2, 0, 0.1])  # [x y z qw qx qy qz]
goalPose4 = np.array([200, 50, 25, 0.7, 0.2, 0, 0.1])
waypoints4 = np.array(
    [
        [0, 100, 45, 0.7, 0.2, 0, 0.1],
        [100, 20, 60, 0.8976, 0.2176, 0.0453, 0.3808],
        [200, 50, 25, 0.7, 0.2, 0, 0.1],
    ]
)

n_waypoints4 = 3
distance4 = np.zeros(n_waypoints4)
for i in range(1, n_waypoints4):
    distance4[i] = np.linalg.norm(waypoints4[i, 0:3] - waypoints4[i - 1, 0:3])

# Assume a UAV speed and calculate time taken to reach each waypoint
UAVspeed4 = 0.835060973068572
timepoints4 = np.cumsum(distance4 / UAVspeed4)
timepoints4[0] = 0  # Make sure the first timepoint is 0
n_samples4 = n_samples

# Compute states along the trajectory
initialStates4, qd4, qdd4, qddd4, qdddd4, pp4, tPoints4, tSamples4 = minsnappolytraj(
    waypoints4.T[0:3, :],
    timepoints4,
    n_samples4,
    min_segment_time=4,
    max_segment_time=60,
)

# Trajectory 5
startPose5 = np.array([200, 200, 35, 0.7, 0.2, 0, 0.1])  # [x y z qw qx qy qz]
goalPose5 = np.array([5, 5, 25, 0.7, 0.2, 0, 0.1])
waypoints5 = np.array(
    [
        [200, 200, 35, 0.7, 0.2, 0, 0.1],
        [100, 100, 60, 0.8976, 0.2176, 0.0453, 0.3808],
        [5, 5, 25, 0.7, 0.2, 0, 0.1],
    ]
)

n_waypoints5 = 3
distance5 = np.zeros(n_waypoints5)
for i in range(1, n_waypoints5):
    distance5[i] = np.linalg.norm(waypoints5[i, 0:3] - waypoints5[i - 1, 0:3])

# Assume a UAV speed and calculate time taken to reach each waypoint
UAVspeed5 = 0.986656316805214
timepoints5 = np.cumsum(distance5 / UAVspeed5)
timepoints5[0] = 0  # Make sure the first timepoint is 0
n_samples5 = n_samples

# Compute states along the trajectory
initialStates5, qd5, qdd5, qddd5, qdddd5, pp5, tPoints5, tSamples5 = minsnappolytraj(
    waypoints5.T[0:3, :],
    timepoints5,
    n_samples5,
    min_segment_time=4,
    max_segment_time=60,
)

# Plot map, start pose, and goal pose
plt.close("all")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.grid(True)
ax.set_xlim([-20, 220])
ax.set_ylim([-20, 220])
ax.set_zlim([-20, 80])


# Plot waypoints
for i in range(n_waypoints):
    ax.scatter(
        waypoints[i, 0],
        waypoints[i, 1],
        waypoints[i, 2],
        c="k",
        marker="o",
        s=30,
        alpha=0.5,
    )
for i in range(n_waypoints2):
    ax.scatter(
        waypoints2[i, 0],
        waypoints2[i, 1],
        waypoints2[i, 2],
        c="r",
        marker="o",
        s=30,
        alpha=0.5,
    )
for i in range(n_waypoints3):
    ax.scatter(
        waypoints3[i, 0],
        waypoints3[i, 1],
        waypoints3[i, 2],
        c="b",
        marker="o",
        s=30,
        alpha=0.5,
    )
for i in range(n_waypoints4):
    ax.scatter(
        waypoints4[i, 0],
        waypoints4[i, 1],
        waypoints4[i, 2],
        c="g",
        marker="o",
        s=30,
        alpha=0.5,
    )
for i in range(n_waypoints5):
    ax.scatter(
        waypoints5[i, 0],
        waypoints5[i, 1],
        waypoints5[i, 2],
        c="y",
        marker="o",
        s=30,
        alpha=0.5,
    )

# Plot start and end positions
ax.scatter(startPose[0], startPose[1], startPose[2], s=100, c="darkred", marker="o")
ax.scatter(goalPose[0], goalPose[1], goalPose[2], s=100, c="darkblue", marker="o")

ax.scatter(startPose2[0], startPose2[1], startPose2[2], s=100, c="darkred", marker="o")
ax.scatter(goalPose2[0], goalPose2[1], goalPose2[2], s=100, c="darkblue", marker="o")

ax.scatter(startPose3[0], startPose3[1], startPose3[2], s=100, c="darkred", marker="o")
ax.scatter(goalPose3[0], goalPose3[1], goalPose3[2], s=100, c="darkblue", marker="o")

ax.scatter(startPose4[0], startPose4[1], startPose4[2], s=100, c="darkred", marker="o")
ax.scatter(goalPose4[0], goalPose4[1], goalPose4[2], s=100, c="darkblue", marker="o")

ax.scatter(startPose5[0], startPose5[1], startPose5[2], s=100, c="darkred", marker="o")
ax.scatter(goalPose5[0], goalPose5[1], goalPose5[2], s=100, c="darkblue", marker="o")

# Create variables to store trajectory lines and UAV markers
(line1,) = ax.plot([], [], [], "k-", linewidth=2)
(line2,) = ax.plot([], [], [], "r-", linewidth=2)
(line3,) = ax.plot([], [], [], "b-", linewidth=2)
(line4,) = ax.plot([], [], [], "g-", linewidth=2)
(line5,) = ax.plot([], [], [], "y-", linewidth=2)

uav_markers = []

# Create a legend for the UAVs
uav1_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="*",
    color="k",
    markerfacecolor="k",
    markersize=10,
    label="UAV1",
)
uav2_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="*",
    color="r",
    markerfacecolor="r",
    markersize=10,
    label="UAV2",
)
uav3_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="*",
    color="b",
    markerfacecolor="b",
    markersize=10,
    label="UAV3",
)
uav4_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="*",
    color="g",
    markerfacecolor="g",
    markersize=10,
    label="UAV4",
)
uav5_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="*",
    color="y",
    markerfacecolor="y",
    markersize=10,
    label="UAV5",
)
start_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="o",
    color="darkred",
    markerfacecolor="darkred",
    markersize=10,
    label="Start",
)
end_marker = plt.Line2D(
    [0],
    [0],
    linestyle="none",
    marker="o",
    color="darkblue",
    markerfacecolor="darkblue",
    markersize=10,
    label="End",
)

ax.legend(
    handles=[
        uav1_marker,
        uav2_marker,
        uav3_marker,
        uav4_marker,
        uav5_marker,
        start_marker,
        end_marker,
    ]
)

# Set up the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("UAV Minimum Snap Trajectories - Progressive Path Visualization")
# ax.view_init(-60, 63)
ax.view_init(elev=20, azim=-60)

# Animation loop - progressively draw the trajectories
for i in range(n_samples):
    # Remove previous UAV markers
    for marker in uav_markers:
        marker.remove()
    uav_markers = []

    # Update line data to show only the traveled path
    line1.set_data(initialStates[0, : i + 1], initialStates[1, : i + 1])
    line1.set_3d_properties(initialStates[2, : i + 1])

    line2.set_data(initialStates2[0, : i + 1], initialStates2[1, : i + 1])
    line2.set_3d_properties(initialStates2[2, : i + 1])

    line3.set_data(initialStates3[0, : i + 1], initialStates3[1, : i + 1])
    line3.set_3d_properties(initialStates3[2, : i + 1])

    line4.set_data(initialStates4[0, : i + 1], initialStates4[1, : i + 1])
    line4.set_3d_properties(initialStates4[2, : i + 1])

    line5.set_data(initialStates5[0, : i + 1], initialStates5[1, : i + 1])
    line5.set_3d_properties(initialStates5[2, : i + 1])

    # Plot current UAV positions
    uav1 = ax.scatter(
        initialStates[0, i],
        initialStates[1, i],
        initialStates[2, i],
        s=100,
        c="k",
        marker="*",
    )
    uav2 = ax.scatter(
        initialStates2[0, i],
        initialStates2[1, i],
        initialStates2[2, i],
        s=100,
        c="r",
        marker="*",
    )
    uav3 = ax.scatter(
        initialStates3[0, i],
        initialStates3[1, i],
        initialStates3[2, i],
        s=100,
        c="b",
        marker="*",
    )
    uav4 = ax.scatter(
        initialStates4[0, i],
        initialStates4[1, i],
        initialStates4[2, i],
        s=100,
        c="g",
        marker="*",
    )
    uav5 = ax.scatter(
        initialStates5[0, i],
        initialStates5[1, i],
        initialStates5[2, i],
        s=100,
        c="y",
        marker="*",
    )

    uav_markers = [uav1, uav2, uav3, uav4, uav5]

    # Update title with progress
    ax.set_title(f"UAV Minimum Snap Trajectories - Step {i + 1}/{n_samples}")

    # Redraw the figure
    plt.draw()
    plt.pause(0.1)

# Keep the final plot visible
plt.tight_layout()
plt.show()
