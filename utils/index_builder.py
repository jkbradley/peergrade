# Class for building indices.

class index_builder(object):
    def __init__(self):
        self.item_list = []  # list of items, in index order
        self.item2index = {} # item --> index

    def __contains__(self, item):
        return item in self.item2index

    # Return index for item.
    # If item is new, create new index.
    def idx(self, item):
        if item in self.item2index:
            return self.item2index[item]
        i = len(self.item_list)
        self.item_list.append(item)
        self.item2index[item] = i
        return i

    # Return item for given index.
    # Return None if not in index.
    def item(self, idx):
        if idx >= len(self.item_list):
            return None
        return self.item_list[idx]

    # Add all items given, in order given.
    def add_all_items(self, items):
        for item in items:
            self.idx(item)


# Static method.  Given list, build index.
def list2index(item_list):
    ib = index_builder()
    ib.add_all_items(item_list)
    return ib.item2index
