from abc import abstractmethod
import os

class DibcoDataItem:
    def __init__(self, in_path, gt_path):
        self.in_path = in_path
        self.gt_path = gt_path

    def __str__(self):
        return 'in_path:%s, gt_path%s'%(self.in_path, self.gt_path)

    def __repr__(self):
        return self.__str__()


class DibcoDataSetIniter:

    def __init__(self, data_root, in_path, gt_path):
        self.data_root = data_root
        self.in_path = os.path.join(self.data_root, in_path)
        self.gt_path = os.path.join(self.data_root, gt_path)
        self.data_list=[]
        fnames = os.listdir(self.in_path)
        for fname in fnames:
            self.data_list.append(DibcoDataItem(in_path = os.path.join(self.in_path, fname),
                                                gt_path = os.path.join(self.gt_path, self.get_tag(fname))))

    @abstractmethod
    def get_tag(self, in_file: str):
        pass



class Dibco2009DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)

class Dibco2010DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2011DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2012DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tif'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2013DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2014DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2016DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        # ps[1] = 'tiff'
        ps[0] = ps[0].replace('_in', '_gt')
        return '.'.join(ps)


class Dibco2017DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        # ps[1] = 'tiff'
        ps[0] += '_gt'
        return '.'.join(ps)


class Dibco2018DataSetIniter(DibcoDataSetIniter):
    def __init__(self, data_root, in_path='in', gt_path='gt'):
        super().__init__(data_root, in_path, gt_path)

    def get_tag(self, in_file: str):
        ps = in_file.split('.')
        # ps[1] = 'tiff'
        ps[0] += '_gt'
        return '.'.join(ps)