import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import csv



number_of_maps = 200
fulllist = []
max_number_of_focii = 20
plantradius = 1
intensity = 200
how_concentrated = 5

path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"


def sample_centre(centre):
    while True: 
        x = np.random.normal(loc = centre[0], scale = how_concentrated)
        y = np.random.normal(loc = centre[1], scale = how_concentrated) 


        if (3.0 < x < 97.0) and (3.0 < y < 97.0):
            return x, y
   

"""
for map in range(number_of_maps + 1):
    with open(rf"{path}\\map_{map}.csv", "w", newline = "") as file:
        writer = csv.writer(file)

        writer.writerow(["map", "x", "y"])
    
        assetposlist = []
        gaussianfocus = [[np.random.uniform(0, 100), np.random.uniform(0, 100)] for _ in range(number_of_focii)]

        for focii in gaussianfocus:
            for i in range(0, 10): 
                assetposlist.append([np.random.normal(loc = focii[0], scale = 5), np.random.normal(loc = focii[1], scale = 5)])

        for _ in range(200): 
            assetposlist.append([np.random.uniform(low = 0, high = 100), np.random.uniform(low = 0, high = 100)])


        assetposlist = np.clip(np.array(assetposlist), np.array([3,3]), np.array([97, 97]))

        for x, y in assetposlist: 
            writer.writerow([map, x, y])
"""

"""
for map in range(number_of_maps + 1):
    with open(rf"{path}\\map_{map}_halffield.csv", "w", newline = "") as file:
        writer = csv.writer(file)

        writer.writerow(["map", "x", "y"])
    
        assetposlist = []
        gaussian_centers = [[np.random.uniform(5, 95), np.random.uniform(65, 95)] for _ in range(number_of_focii)]

        for center in gaussian_centers:
            for _ in range(10):
                assetposlist.append([
                    np.random.normal(loc=center[0], scale=5),
                    np.random.normal(loc=center[1], scale=5)
                ])

        # original single large blob, kept commented out for reference
        # center = [np.random.uniform(5, 95), np.random.uniform(5, 95)]
        # for i in range(0, 100):
        #     assetposlist.append([
        #         np.random.normal(loc=center[0], scale=20),
        #         np.random.normal(loc=center[1], scale=20)
        #     ])

        assetposlist = np.clip(np.array(assetposlist), np.array([3, 3]), np.array([97, 97]))

        for x, y in assetposlist:
            writer.writerow([map, x, y])
"""

for map in range(number_of_maps + 1):
    with open(rf"{path}\\map_{map}_multiblob.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["map", "x", "y"])

        assetposlist = []
        # Multimodal expert paths need SEVERAL EQUAL, WELL-SEPARATED targets so the ORDER the
        # planner visits them in is a genuine free choice (many paths -> same variance reduction).
        # Two simple properties do it:
        #   (a) few blobs (not a dense mass), each with the SAME point count -> no dominant blob
        #       that the planner would always resolve first, and
        #   (b) a minimum spacing between blob centres -> distinct, separated targets rather than
        #       one merged region, so "which target next" stays a real choice at every step.
        number_of_blobs = np.random.randint(8, max_number_of_focii)  # 8-20 distinct targets
        min_separation = 12.0                     # keep centres this far apart (map units)
        gaussian_centers = []
        attempts = 0
        while len(gaussian_centers) < number_of_blobs and attempts < 1000:
            attempts += 1
            candidate = [np.random.uniform(15, 85), np.random.uniform(15, 85)]
            if all(np.hypot(candidate[0] - cx, candidate[1] - cy) >= min_separation
                   for cx, cy in gaussian_centers):
                gaussian_centers.append(candidate)

        for center in gaussian_centers:
            for _ in range(intensity):  # SAME intensity for every blob => equal-value targets
                x, y = sample_centre(centre=center)
                assetposlist.append([x, y])

        # assetposlist = np.clip(np.array(assetposlist), np.array([3, 3]), np.array([97, 97]))

        for x, y in assetposlist:
            writer.writerow([map, x, y])


        
        
        






    

fig, ax = plt.subplots()

for x, y in assetposlist:
    circle = Circle((x,y), radius = plantradius, alpha = 0.5)
    ax.add_patch(circle)


# # ax.scatter([x for x, y in gaussianfocus],
# #            [y for x, y in gaussianfocus],
# #            marker='x', s=100)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal') 

ax.set_xlabel("meters")
ax.set_ylabel("meters")

plt.show()

# plt.scatter([assetcoord[0] for assetcoord in assetposlist], [assetcoord[1] for assetcoord in assetposlist])
# plt.scatter([focii[0] for focii in gaussianfocus], [focii[1] for focii in gaussianfocus])
plt.show()
