import numpy as np
import matplotlib.pyplot as plt

from MinMpc2_WithoutTC import MinMpc2_WithoutTC

import matlab.engine

# pip install matlabengine==24.2.2
eng = matlab.engine.start_matlab()


def main():
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

    initialStates, qd, qdd, qddd, qdddd, pp, tPoints, tSamples = eng.minsnappolytraj(
        waypoints.T,
        timepoints,
        n_samples,
        "MinSegmentTime",
        12,
        "MaxSegmentTime",
        60,
        nargout=8,
    )

    tSamples = np.array(tSamples)

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

    initialStates2, qd2, qdd2, qddd2, qdddd2, pp2, tPoints2, tSamples2 = (
        eng.minsnappolytraj(
            waypoints2.T,
            timepoints2,
            n_samples2,
            "MinSegmentTime",
            12,
            "MaxSegmentTime",
            60,
            nargout=8,
        )
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
    initialStates3, qd3, qdd3, qddd3, qdddd3, pp3, tPoints3, tSamples3 = (
        eng.minsnappolytraj(
            waypoints3.T,
            timepoints3,
            n_samples3,
            "MinSegmentTime",
            12,
            "MaxSegmentTime",
            60,
            nargout=8,
        )
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
    initialStates4, qd4, qdd4, qddd4, qdddd4, pp4, tPoints4, tSamples4 = (
        eng.minsnappolytraj(
            waypoints4.T,
            timepoints4,
            n_samples4,
            "MinSegmentTime",
            12,
            "MaxSegmentTime",
            60,
            nargout=8,
        )
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
    initialStates5, qd5, qdd5, qddd5, qdddd5, pp5, tPoints5, tSamples5 = (
        eng.minsnappolytraj(
            waypoints5.T,
            timepoints5,
            n_samples5,
            "MinSegmentTime",
            12,
            "MaxSegmentTime",
            60,
            nargout=8,
        )
    )

    # Define early start times
    EarlyStart1 = 25
    EarlyStart2 = 50

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True)
    ax.set_xlim([-20, 220])
    ax.set_ylim([-20, 220])
    ax.set_zlim([-20, 80])

    # Plot starting and goal points
    ax.scatter(startPose[0], startPose[1], startPose[2], color="m", s=50, marker="o")
    ax.scatter(goalPose[0], goalPose[1], goalPose[2], color="g", s=50, marker="o")

    ax.scatter(startPose2[0], startPose2[1], startPose2[2], color="m", s=50, marker="o")
    ax.scatter(goalPose2[0], goalPose2[1], goalPose2[2], color="g", s=50, marker="o")

    ax.scatter(startPose3[0], startPose3[1], startPose3[2], color="m", s=50, marker="o")
    ax.scatter(goalPose3[0], goalPose3[1], goalPose3[2], color="g", s=50, marker="o")

    ax.scatter(startPose4[0], startPose4[1], startPose4[2], color="m", s=50, marker="o")
    ax.scatter(goalPose4[0], goalPose4[1], goalPose4[2], color="g", s=50, marker="o")

    ax.scatter(startPose5[0], startPose5[1], startPose5[2], color="m", s=50, marker="o")
    ax.scatter(goalPose5[0], goalPose5[1], goalPose5[2], color="g", s=50, marker="o")

    plt.draw()
    plt.pause(0.01)

    # Trajectory visualization for the first trajectory
    ActualtSamples1 = tSamples[tSamples < EarlyStart1]
    ll = len(ActualtSamples1)
    ActualStates1 = eng.ppval(pp, ActualtSamples1)
    ActualStates1 = np.array(ActualStates1)

    for i in range(ll):
        ax.plot(
            ActualStates1[0, i],
            ActualStates1[1, i],
            ActualStates1[2, i],
            ".m",
            markersize=10,
        )
        plt.draw()
        plt.pause(0.2)
        # ax.view_init(-61, 63)
        ax.view_init(elev=20, azim=-60)


    # MPC coordination for 2 UAVs
    n = 2
    L = np.array([[1, -1], [-1, 1]])
    N = 20
    step = 0.05
    gammadot = np.ones((n, N))
    gamma = np.vstack(
        [
            EarlyStart1 + gammadot[0, 0] * step * np.arange(N),
            gammadot[1, 0] * step * np.arange(N),
        ]
    )

    u = np.empty((n, 0))
    t = EarlyStart1
    hh = 0

    newpoint1 = eng.ppval(pp, gamma[0, -1])
    newpoint2 = eng.ppval(pp2, gamma[1, -1])

    L = np.array([[1, -1], [-1, 1]])

    # First coordination phase (2 UAVs)
    while t < EarlyStart2:
        if hh % 20 == 0:
            ax.plot(newpoint1[0], newpoint1[1], newpoint1[2], ".c", markersize=10)
            ax.plot(newpoint2[0], newpoint2[1], newpoint2[2], ".c", markersize=10)
            plt.draw()
            plt.pause(0.05)
            # ax.view_init(-61, 63)
            ax.view_init(elev=20, azim=-60)

        value, derivative, uu = MinMpc2_WithoutTC(
            L, gamma[:, -(N):], gammadot[:, -(N):], step, N
        )
        gamma = np.hstack([gamma[:, : -(N - 1)], value])
        gammadot = np.hstack([gammadot[:, : -(N - 1)], derivative])
        u = np.hstack([u, uu[:, 0].reshape(-1, 1)])

        newpoint1 = eng.ppval(pp, gamma[0, -(N)])
        newpoint2 = eng.ppval(pp2, gamma[1, -(N)])

        hh += 1
        t += step

    # MPC coordination for 5 UAVs
    n = 5
    gammadot_new = np.vstack(
        [gammadot[:, -(N):], np.ones((1, N)), np.ones((1, N)), np.ones((1, N))]
    )

    gamma_new = np.vstack(
        [
            gamma[:, -(N):],
            gammadot_new[2, 0] * step * np.arange(N),
            gammadot_new[3, 0] * step * np.arange(N),
            gammadot_new[4, 0] * step * np.arange(N),
        ]
    )

    gamma = gamma_new
    gammadot = gammadot_new
    u = np.empty((n, 0))
    t = EarlyStart2

    newpoint3 = eng.ppval(pp3, gamma[2, -(N)])
    newpoint4 = eng.ppval(pp4, gamma[3, -(N)])
    newpoint5 = eng.ppval(pp5, gamma[4, -(N)])

    # Set up the Laplacian matrix for 5 UAVs
    L = np.array(
        [
            [4, -1, -1, -1, -1],
            [-1, 4, -1, -1, -1],
            [-1, -1, 4, -1, -1],
            [-1, -1, -1, 4, -1],
            [-1, -1, -1, -1, 4],
        ]
    )

    # Second coordination phase (5 UAVs)
    while (gamma[0, -1] < tSamples[-1]).any():
        if hh % 40 == 0:
            ax.plot(newpoint1[0], newpoint1[1], newpoint1[2], ".k", markersize=8)
            ax.plot(newpoint2[0], newpoint2[1], newpoint2[2], ".r", markersize=8)
            ax.plot(newpoint3[0], newpoint3[1], newpoint3[2], ".b", markersize=8)
            ax.plot(newpoint4[0], newpoint4[1], newpoint4[2], ".g", markersize=8)
            ax.plot(newpoint5[0], newpoint5[1], newpoint5[2], ".y", markersize=8)
            plt.draw()
            plt.pause(0.03)
            # ax.view_init(-61, 63)
            ax.view_init(elev=20, azim=-60)


        value, derivative, uu = MinMpc2_WithoutTC(
            L, gamma[:, -(N):], gammadot[:, -(N):], step, N
        )
        gamma = np.hstack([gamma[:, : -(N - 1)], value])
        gammadot = np.hstack([gammadot[:, : -(N - 1)], derivative])
        u = np.hstack([u, uu[:, 0].reshape(-1, 1)])

        newpoint1 = eng.ppval(pp, gamma[0, -(N)])
        newpoint2 = eng.ppval(pp2, gamma[1, -(N)])
        newpoint3 = eng.ppval(pp3, gamma[2, -(N)])
        newpoint4 = eng.ppval(pp4, gamma[3, -(N)])
        newpoint5 = eng.ppval(pp5, gamma[4, -(N)])

        hh += 1
        t += step

    plt.show()

    ###########################
    ###########################
    # Plotting gamma, gammadot, and u values
    ###########################
    ###########################

    # Create time vectors for each plot
    time_gamma = np.arange(0, step * (len(gamma[0, :-N]) + 1), step)
    time_u = np.arange(0, step * len(u[0, :-1]), step)

    # Define colors for consistency with MATLAB
    colors = [
        "k",
        "r",
        "b",
        "g",
        "#edaa3a",
    ]  # Last color approximates [0.9290, 0.6940, 0.1250]

    # Figure 3: Plot gamma values
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(
            time_gamma[: len(gamma[i, : -(N - 1)])],
            gamma[i, : -(N - 1)],
            color=colors[i],
            linewidth=1.5,
        )
    plt.legend(["UAV1", "UAV2", "UAV3", "UAV4", "UAV5"])
    plt.title("Gamma Values")
    plt.grid(True)
    plt.show()

    # Figure 4: Plot gammadot values
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(
            time_gamma[: len(gammadot[i, : -(N - 1)])],
            gammadot[i, : -(N - 1)],
            color=colors[i],
            linewidth=1.5,
        )
    plt.legend(["UAV1", "UAV2", "UAV3", "UAV4", "UAV5"])
    plt.title("Gamma Derivatives")
    plt.grid(True)
    plt.show()

    # Figure 5: Plot u values
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(time_u[: len(u[i, :])], u[i, :], color=colors[i], linewidth=1.5)
    plt.legend(["UAV1", "UAV2", "UAV3", "UAV4", "UAV5"])
    plt.title("Control Inputs")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
