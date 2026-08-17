def target_encoding(categories, targets):
    category_stats = {}
    
    for cat, target in zip(categories, targets):
        if cat not in category_stats:
            category_stats[cat] = {'sum': 0, 'count': 0}
        category_stats[cat]['sum'] += target
        category_stats[cat]['count'] += 1
        
    category_means = {
        cat: stats['sum'] / stats['count'] 
        for cat, stats in category_stats.items()
    }
    
    return [category_means[cat] for cat in categories]
