normal_label = cat_dict['Normal']

train_normal_df = df1.loc[df1['label'] == normal_label]

train_X = df1.values[:, :-1]
train_normal = train_normal_df.values[:, :-1]
train_Y = df1.values[:, -1]

train_Y_binary = train_Y == normal_label
train_Y = np.array(train_Y_binary).astype(np.int64)

train_Y = np.array(train_Y).astype(np.int64)