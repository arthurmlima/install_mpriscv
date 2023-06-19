from pathlib import Path


class PathResolver(object):

    @staticmethod
    def project():
        return Path(__file__).parents[2]

    @staticmethod
    def project_parent():
        return PathResolver.project().parent

    @staticmethod
    def module():
        return PathResolver.project() / 'red1000_deng'

    @staticmethod
    def images():
        return PathResolver.codes() / 'images/'

    @staticmethod
    def data_folder():
        return PathResolver.codes() / 'data/'

    @staticmethod
    def water_images():
        return PathResolver.codes() / 'images/SP/'

    @staticmethod
    def codes():
        return Path(__file__).parents[1]

    @staticmethod
    def codes():
        return Path(__file__).parents[1]
