import json


def load_json():
    path = 'multiprocess_handler/test_project.json'
    path = '../multiprocess_handler/test_project.json'

    with open(path) as json_file:
        return json.load(json_file)


config = load_json()
#
# config = {
#     "directory": "/home/user/KaelFacon/Project/rgbd_mocap/data_files/P4_session2/gear_20_15-08-2023_10_52_14",
#     "start_index": 1515,
#     "end_index": 3189,
#     "crops": [
#         {
#             "name": "Back",
#             "area": [
#                 340,
#                 141,
#                 481,
#                 380
#             ],
#             "filters": {
#                 "blend": 100,
#                 "white_range": [
#                     137,
#                     255
#                 ],
#                 "blob_area": [
#                     23,
#                     205
#                 ],
#                 "convexity": 43,
#                 "circularity": 44,
#                 "distance_between_blobs": 15,
#                 "distance_in_centimeters": [
#                     12,
#                     98
#                 ],
#                 "clahe_clip_limit": 3,
#                 "clahe_grid_size": 5,
#                 "gaussian_blur": 2,
#                 "use_contour": True,
#                 'mask': None,
#                 "white_option": True,
#                 "blob_option": True,
#                 "clahe_option": True,
#                 "distance_option": True,
#                 "masks_option": True
#             },
#             "markers": [
#                 {
#                     "name": "g",
#                     "pos": [
#                         69,
#                         121
#                     ]
#                 },
#                 {
#                     "name": "j",
#                     "pos": [
#                         79,
#                         108
#                     ]
#                 },
#                 {
#                     "name": "k",
#                     "pos": [
#                         148,
#                         107
#                     ]
#                 },
#                 {
#                     "name": "c",
#                     "pos": [
#                         212,
#                         60
#                     ]
#                 },
#                 {
#                     "name": "a",
#                     "pos": [
#                         91,
#                         34
#                     ]
#                 },
#                 {
#                     "name": "c",
#                     "pos": [
#                         65,
#                         35
#                     ]
#                 },
#                 {
#                     "name": "o",
#                     "pos": [
#                         10,
#                         95
#                     ]
#                 }
#             ]
#         },
#         {
#             "name": "Crop",
#             "area": [
#                 184,
#                 227,
#                 349,
#                 366
#             ],
#             "filters": {
#                 "blend": 100,
#                 "white_range": [
#                     189,
#                     255
#                 ],
#                 "blob_area": [
#                     12,
#                     65
#                 ],
#                 "convexity": 10,
#                 "circularity": 10,
#                 "distance_between_blobs": 10,
#                 "distance_in_centimeters": [
#                     59,
#                     102
#                 ],
#                 "clahe_clip_limit": 1,
#                 "clahe_grid_size": 1,
#                 "gaussian_blur": 2,
#                 "use_contour": True,
#                 "mask": None,
#                 "white_option": True,
#                 "blob_option": True,
#                 "clahe_option": True,
#                 "distance_option": True,
#                 "masks_option": False
#             },
#             "markers": [
#                 {
#                     "name": "a",
#                     "pos": [
#                         34,
#                         15
#                     ]
#                 },
#                 {
#                     "name": "b",
#                     "pos": [
#                         16,
#                         27
#                     ]
#                 },
#                 {
#                     "name": "c",
#                     "pos": [
#                         58,
#                         44
#                     ]
#                 }
#             ]
#         },
#         {
#             "name": "Crop",
#             "area": [
#                 251,
#                 271,
#                 362,
#                 404
#             ],
#             "filters": {
#                 "blend": 100,
#                 "white_range": [
#                     147,
#                     242
#                 ],
#                 "blob_area": [
#                     10,
#                     100
#                 ],
#                 "convexity": 10,
#                 "circularity": 10,
#                 "distance_between_blobs": 10,
#                 "distance_in_centimeters": [
#                     20,
#                     102
#                 ],
#                 "clahe_clip_limit": 4,
#                 "clahe_grid_size": 1,
#                 "gaussian_blur": 5,
#                 "use_contour": True,
#                 "mask": None,
#                 "white_option": True,
#                 "blob_option": True,
#                 "clahe_option": True,
#                 "distance_option": True,
#                 "masks_option": False
#             },
#             "markers": [
#                 {
#                     "name": "a",
#                     "pos": [
#                         34,
#                         15
#                     ]
#                 },
#                 {
#                     "name": "b",
#                     "pos": [
#                         16,
#                         27
#                     ]
#                 },
#                 {
#                     "name": "c",
#                     "pos": [
#                         58,
#                         44
#                     ]
#                 }
#             ]
#         }
#     ],
#     "markers": []
# }
