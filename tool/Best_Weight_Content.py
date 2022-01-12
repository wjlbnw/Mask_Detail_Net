import torch
import os

class Best_Weights_Content_Item:
    def __init__(self, score, weight_path):
        self.score = score
        # self.weight_dict = weigth_dict
        self.weight_path = weight_path


class Best_Weights_Content:

    def __init__(self, max_len):
        self.max_len = max_len
        self.item_dict = {}
        self.key_constructor = 0

    def get_key(self):
        key = self.key_constructor
        self.key_constructor += 1
        return key
    def insert(self, score, weigth_dict, weight_path):
        if (len(self.item_dict.keys()) < self.max_len):
            key = self.get_key()
            self.item_dict[key] = Best_Weights_Content_Item(score, weight_path)
            torch.save(weigth_dict, weight_path)
        else:
            min_key = -1
            for k in self.item_dict.keys():
                if min_key == -1:
                    min_key = k
                elif self.item_dict[k].score < self.item_dict[min_key].score:
                    min_key = k

            if min_key != -1 and self.item_dict[min_key].score < score:
                os.remove(self.item_dict[min_key].weight_path)
                del self.item_dict[min_key]
                key = self.get_key()
                self.item_dict[key] = Best_Weights_Content_Item(score, weight_path)
                torch.save(weigth_dict, weight_path)