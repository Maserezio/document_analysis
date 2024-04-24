import doxapy
from tabulate import tabulate

def evaluate(gt_imgs, pr_imgs, print_results=True):
    perf_data = []

    for gt_img, pr_img in zip(gt_imgs, pr_imgs):
        perf_data.append(doxapy.calculate_performance(gt_img, pr_img))

    mean_values = {}

    for d in perf_data:
        for key, value in d.items():
            if key not in mean_values:
                mean_values[key] = []
            mean_values[key].append(value)
        
    nice_names = {
        'accuracy': 'Accuracy (%)',
        'fm': 'F-measure',
        'mcc': 'Matthews Correlation Coefficient',
        'psnr': 'Peak Signal-to-Noise Ratio (PSNR)',
        'nrm': 'Normalized Root Mean Square Error (NRM)',
        'drdm': 'Distance-based Performance Measure (DRDM)'
    }

    mean_values_single = {nice_names[key]: round(value[0], 2) for key, value in mean_values.items()}
    
    if(print_results == True):
        data = [
            ["Metric", "Processed Images"],
            ["Accuracy (%)", "{:.2f}".format(mean_values_single['Accuracy (%)'])],
            ["F-measure", "{:.2f}".format(mean_values_single['F-measure'])],
            ["Matthews Correlation Coefficient", "{:.2f}".format(mean_values_single['Matthews Correlation Coefficient'])],
            ["Peak Signal-to-Noise Ratio (PSNR)", "{:.2f}".format(mean_values_single['Peak Signal-to-Noise Ratio (PSNR)'])],
            ["Normalized Root Mean Square Error (NRM)", "{:.2f}".format(mean_values_single['Normalized Root Mean Square Error (NRM)'])],
            ["Distance-based Performance Measure (DRDM)", "{:.2f}".format(mean_values_single['Distance-based Performance Measure (DRDM)'])],
        ]

        print(tabulate(data, headers="firstrow", tablefmt="grid"))

    return mean_values_single

def show_metrics_table(metrics_dicts):
    metrics_names = list(metrics_dicts.values())[0].keys()

    headers = ["Metric"] + list(metrics_dicts.keys())

    table_data = []

    for metric_name in metrics_names:
        row_data = [metric_name]
        for metrics_dict in metrics_dicts.values():
            row_data.append("{:.2f}".format(metrics_dict[metric_name]))
        table_data.append(row_data)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))