import paddle.fluid as fluid
import numpy 

x = fluid.data(name='x', shape=[2, 4], dtype='float64')

reduce_sum_no_dim = fluid.layers.reduce_sum(x)
reduce_sum_empty_dim = fluid.layers.reduce_sum(x, dim = [])

reduce_mean_no_dim = fluid.layers.reduce_mean(x)
reduce_mean_empty_dim = fluid.layers.reduce_mean(x,dim = [])

reduce_max_no_dim = fluid.layers.reduce_max(x)
reduce_max_empty_dim = fluid.layers.reduce_max(x, dim = [])

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[x], place=place)
exe.run(fluid.default_startup_program())

name_list = [reduce_sum_no_dim,reduce_sum_empty_dim,reduce_mean_no_dim, reduce_mean_empty_dim, reduce_max_no_dim, reduce_max_empty_dim]
data = [[numpy.random.random((2,4)).astype("float64")]]

print(data)

sum_no, sum_empty, mean_no, mean_empty, max_no, max_empty = exe.run(
                                                              fluid.default_main_program(),
                                                              feed=feeder.feed(data),
                                                              fetch_list = name_list,
                                                              return_numpy=True)


print(sum_no)
print(sum_empty)
print(mean_no)
print(mean_empty)
print(max_no)
print(max_empty)
