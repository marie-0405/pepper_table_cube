import numpy as np
import matplotlib.pyplot as plt
from hyper_parameter import HyperParameter
from result_controller import ResultController


class HyperParameterController():
  def __init__(self, alphas, gammas) -> None:
    self.alphas = alphas
    self.gammas = gammas
    
  def _create_hyperparams_grid(self):
    graph_x = []
    graph_y = []
    graph_z = []
    for alpha in self.alphas:
        graph_x_row = []
        graph_y_row = []
        graph_z_row = []
        for gamma in self.gammas:
          result_controller = ResultController("a={:.1}-g={:.1}".format(alpha, gamma))
          average = result_controller.get_average()
          positive_average = average + 20
          graph_x_row.append(alpha)
          graph_y_row.append(gamma)
          graph_z_row.append(positive_average)        
        graph_x.append(graph_x_row)
        graph_y.append(graph_y_row)
        graph_z.append(graph_z_row)
    graph_x=np.array(graph_x)
    graph_y=np.array(graph_y)
    graph_z=np.array(graph_z)
    min_z = np.min(graph_z)
    pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
    print('Minimum RMSE: %.4f' %(min_z))
    print('Optimum alpha: %f' %(graph_x[pos_min_z[0],pos_min_z[1]]))
    print('Optimum gamma: %f' %(graph_y[pos_min_z[0],pos_min_z[1]]))
    return graph_x,graph_y,graph_z

  def plot_hyperparams_grid(self):
    graph_x, graph_y, graph_z = self._create_hyperparams_grid()
    size_list=np.array(graph_z)
    size_list=size_list**2 * 4
    points=plt.scatter(graph_x, graph_y, c=graph_z, cmap='viridis',vmin=5,vmax=11,marker='o',s=size_list)
    cbar=plt.colorbar(points)
    cbar.set_label("$AVERAGE$", fontsize=14)
    plt.xlabel(r'$\alpha$',fontsize=14)
    plt.ylabel(r'$\gamma$',fontsize=14)
    plt.yticks(np.arange(0.3,1.0,0.1))
    file_name = 'hyperparams_grid.png'
    plt.savefig(file_name,format='png',dpi=600)
    plt.close()

if __name__ == '__main__':
  Alphas = HyperParameter(0.2, 0.8, 0.1)
  Gammas = HyperParameter(0.5, 0.7, 0.1)
  # print(Alphas.values, Gammas.values)  
  hyper_parameter_controller = HyperParameterController(Alphas.values, Gammas.values)
  hyper_parameter_controller.plot_hyperparams_grid()