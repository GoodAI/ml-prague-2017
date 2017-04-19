import matplotlib.pyplot as plot


def plot_stats(name, multiple_stats):
    plot.gcf().clear()
    plot.ylabel('accuracy')
    plot.xlabel('episode')
    handles = [plot.plot(stats.labels, stats.accuracy, label=stats.name)
               for stats in multiple_stats
               if stats is not None]

    plot.legend(handles=[handle[0] for handle in handles], loc='lower right')
    plot.savefig(name + '_accuracy.png')
