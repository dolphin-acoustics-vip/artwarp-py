
def build_charts(results):
    iterations, reclassified = zip(*results.iteration_history)
    cat_size_dic = results.get_category_sizes()
    categories = cat_size_dic.keys()
    sizes = cat_size_dic.values()

    convergence = {
        "kind": "line",
        "title": "Convergence",
        "desc": "Contours that changed category in each iteration. Zero means converged.",
        "x": list(iterations),
        "x_label": "Iteration",
        "y_label": "Reclassifications",
        "y_format": "int",
        "y_zero": True,
        "series": [
            {"name": "Reclassified", "slot": 0, "values": list(reclassified)},
        ],
    }

    bar_sizes = {
        "kind": "bar", 
        "title": "Category Sizes", 
        "desc": "Size of each category of contours", 
        "categories": [str(cat+1) for cat in list(categories)],
        "x_label": "Categories", 
        "y_label": "Contours", 
        "y_format": "int",
        "series": [
            {"name": "Size", "slot": 0, "values": list(sizes)},
        ]
    }   

    return [convergence, bar_sizes]

