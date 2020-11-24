# leaf_to_label_dict = dict()
# leaf_to_soft_label_dict = dict()
# for i in range(len(leaf_pred)):
#     leaf_to_label_dict.setdefault(leaf_pred[i], []).append(labels[i])
#
# for k, v in leaf_to_label_dict.items():
#     v_d = dict()
#     vu, vc = np.unique(np.array(v), return_counts=True)
#     for i in range(len(vu)):
#         v_d[vu[i]] = vc[i]
#
#     leaf_to_label_dict[k] = v_d
#
# pprint.PrettyPrinter(indent=4, width=180).pprint(leaf_to_label_dict)
#
# leaf_pred = clf.apply(test_dataset.get_x())
# print(len(leaf_pred))
#
# leaf_to_label_dict = dict()
# for i in range(len(leaf_pred)):
#     leaf_to_label_dict.setdefault(leaf_pred[i], []).append(test_dataset.get_y()[i])
#
# for k, v in leaf_to_label_dict.items():
#     v_d = dict()
#     vu, vc = np.unique(np.array(v), return_counts=True)
#     for i in range(len(vu)):
#         v_d[vu[i]] = vc[i]
#
#     leaf_to_label_dict[k] = v_d
#
# pprint.PrettyPrinter(indent=4, width=180).pprint(leaf_to_label_dict)
#
# cluster_assignment = clustering.predict(test_dataset.get_x())
#
# leaf_to_label_dict = dict()
# for i in range(len(cluster_assignment)):
#     leaf_to_label_dict.setdefault(cluster_assignment[i], []).append(test_dataset.get_y()[i])
#
# for k, v in leaf_to_label_dict.items():
#     v_d = dict()
#     vu, vc = np.unique(np.array(v), return_counts=True)
#     for i in range(len(vu)):
#         v_d[vu[i]] = vc[i]
#
#     leaf_to_label_dict[k] = v_d
#
# for k, v in cluster_to_label_dict.items():
#     v_d = dict()
#     vu, vc = np.unique(np.array(v), return_counts=True)
#     for i in range(len(vu)):
#         v_d[vu[i]] = vc[i]
#
#     cluster_to_label_dict[k] = v_d
#
# pprint.PrettyPrinter(indent=4, width=180).pprint(leaf_to_label_dict)
# pprint.PrettyPrinter(indent=4, width=180).pprint(cluster_to_label_dict)
#
# cl_centers = clustering.cluster_centers_
#
# cl_center_distances = np.zeros((len(cl_centers), len(cl_centers)))
# for i in range(len(cl_centers)):
#     for j in range(i + 1, len(cl_centers)):
#         cl_center_distances[i][j] = np.linalg.norm(cl_centers[i] - cl_centers[j])
#         cl_center_distances[j][i] = cl_center_distances[i][j]
#
# nearest = dict()
# for i in range(len(cl_centers)):
#     dist_sort = np.argsort(cl_center_distances[i])
#     nearest[i] = dist_sort[:20]
#
# pprint.PrettyPrinter(indent=4, width=200).pprint(nearest)
