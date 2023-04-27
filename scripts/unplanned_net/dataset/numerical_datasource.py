from unplanned_net.dataset.datasource import NumericalDataSource, register_datasource

@register_datasource("age_at_adm")
class AgeDataSource(NumericalDataSource):
    def __init__(self, stats:dict =  None):
        self.vocab = None
        self.stats = stats
        self.name = 'age_at_adm'
        print (f"Datasource {self.name} running !")

@register_datasource("idx_adm")
class IndexAdmissionDataSource(NumericalDataSource):
    def __init__(self, stats:dict =  None):
        self.vocab = None
        self.stats = stats
        self.name = 'idx_adm'
        print (f"Datasource {self.name} running !")

@register_datasource("sex")
class SexDataSource(NumericalDataSource):
    def __init__(self, stats:dict =  None):
        self.vocab = None
        self.stats = stats
        self.name = 'sex'
        print (f"Datasource {self.name} running !")
