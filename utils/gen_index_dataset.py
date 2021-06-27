from torch.utils.data import Dataset


class gen_index_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image, each_label, each_true_label, index

    '''
    def __init__(self, user_input, item_input, ratings):
        self.user_input = user_input
        self.item_input = item_input
        self.ratings = ratings

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, index):
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        rating = self.ratings[index]

        return {'user_id': user_id,
                'item_id': item_id,
                'rating': rating}
    '''
