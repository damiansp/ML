from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.graphskeleton import GraphSkeleton
from libpgm.nodedata import NodeData
from libpgm.tablecpdfactorization import TableCPDFactorization


def get_table_cpd():
    nd = NodeData()
    skel = GraphSkeleton()
    json_path = 'job_interview.txt'
    nd.load(json_path)
    skel.load(json_path)
    bn = DiscreteBayesianNetwork(skel, nd)
    table_cpd = TableCPDFactorization(bn)
    return table_cpd

table = get_table_cpd()
print(table)

print(table.specificquery(dict(Offer='1'), dict()))
print(table.specificquery(dict(Offer='1'), dict(Grades='0')))
print(table.specificquery(dict(Experience='1'), dict()))
print(table.specificquery(dict(Experience='1'), dict(Interview='2')))
print(
    table.specificquery(dict(Experience='1'), dict(Interview='2', Grades='0')))

      
