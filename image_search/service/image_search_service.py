from abc import ABC, abstractmethod


class ImageSearchService(ABC):
    @abstractmethod
    def imageSearch(self,query_image_path, top_k):
        pass
    @abstractmethod
    def search(self, query_image_path):
        pass