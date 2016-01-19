import numpy as np
import qt
import plot

x = np.arange(-10, 10, 0.2)
y = np.sinc(x)
yerr = 0.1*np.random.rand(len(x))

d = qt.Data()
d.add_coordinate('x')
d.add_coordinate('y')
d.add_coordinate('yerr')
d.create_file()
d.add_data_point(x,y,yerr)

p = plot.get_plot('error bar test plot',
                  replace_if_exists=True)
p = p.get_plot()
p.add_trace(d['x'], d['y'], yerr=d['yerr'])
p.update()
#.run() # necessary if you've closed the window
