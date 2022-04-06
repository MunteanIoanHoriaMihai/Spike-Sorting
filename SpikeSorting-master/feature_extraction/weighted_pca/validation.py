import csv

from main_aldea import do_spike_sorting


def validate_method_all():
    """
    Performs spike sorting for all the three cases:
    1. when applying only PCA 2D
    2. when applying PCA 2D and weighting considering the peaks
    3. when applying PCA 2D and weighting considering both peaks and divergence

    and for all three clustering cases:
    1. on ground truth
    2. when applying K-Means
    3. when applying SBM

    A subset of simulations are used stores in sims 1D array.
    Results are stored in .csv files. For each algorithm there should be a different .csv file.

    When evaluating on ground truth, only internal validation is applied.
    """

    filename = './validation_all/SBM_new_method_1015.csv'

    sims = [1, 2, 3, 4, 5, 6, 7, 8, 14, 16, 18, 21, 22, 24, 29, 30, 31, 32, 33, 39, 42, 46, 48, 53, 54, 57, 58, 59, 62,
            64, 76, 77, 81,
            83, 84, 87, 88, 89, 93, 94]

    algs = ['', 'K-Means', 'SBM']
    row = []

    for sim in sims:
        print(sim)
        for i, a in enumerate(algs):
            internal_1, external_1, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                            weighted_method=False,
                                                            nr_features=79, do_plot=False, do_save=False,
                                                            folder='validation/sim' + str(sim), alg=a,
                                                            scaling=False,
                                                            minmax=True,
                                                            divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                                sim) + '_base2.txt',
                                                            use_divergence=False,
                                                            div_method='js_div'
                                                            )

            internal_2, external_2, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                            weighted_method=True,
                                                            nr_features=79, do_plot=False, do_save=False,
                                                            folder='validation/sim' + str(sim), alg=a,
                                                            scaling=False,
                                                            minmax=True,
                                                            divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                                sim) + '_base2.txt',
                                                            use_divergence=False,
                                                            div_method='js_div'
                                                            )

            internal_3, external_3, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                            weighted_method=True,
                                                            nr_features=79, do_plot=False, do_save=False,
                                                            folder='validation/sim' + str(sim), alg=a,
                                                            scaling=False,
                                                            minmax=True,
                                                            divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                                sim) + '_base2.txt',
                                                            use_divergence=True,
                                                            div_method='js_div'
                                                            )

            # if i == 0:
            #     row = [[sim] +
            #            [internal_1[0]] + [internal_1[1]] + [internal_1[2]] +
            #            [internal_2[0]] + [internal_2[1]] + [internal_2[2]] +
            #            [internal_3[0]] + [internal_3[1]] + [internal_3[2]]]

            # row = [[sim] +
            #        [external_1[0]] + [external_1[1]] + [internal_1[0]] + [internal_1[1]] + [internal_1[2]] +
            #        [external_2[0]] + [external_2[1]] + [internal_2[0]] + [internal_2[1]] + [internal_2[2]] +
            #        [external_3[0]] + [external_3[1]] + [internal_3[0]] + [internal_3[1]] + [internal_3[2]]]

            row = [[sim] +
                   [external_1[0]] + [external_1[1]] + [external_1[2]] + [external_1[3]] + [internal_1[0]] + [
                       internal_1[1]] + [internal_1[2]] +
                   [external_2[0]] + [external_2[1]] + [external_2[2]] + [external_2[3]] + [internal_2[0]] + [
                       internal_2[1]] + [internal_2[2]] +
                   [external_3[0]] + [external_3[1]] + [external_3[2]] + [external_3[3]] + [internal_3[0]] + [
                       internal_3[1]] + [internal_3[2]]]

            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(row)


def validate_KMeans_SBM():
    filename = './validation_all/gt_pca2d_.csv'

    algs = ['', 'K-Means', 'SBM']
    row = []
    rows = []

    for sim in range(1, 95):
        print(sim)
        if sim != 25 and sim != 44:
            for i, a in enumerate(algs):
                internal_1, external_1, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                                weighted_method=False,
                                                                nr_features=79, do_plot=False, do_save=False,
                                                                folder='validation/sim' + str(sim), alg=a,
                                                                scaling=False,
                                                                minmax=True,
                                                                divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                                    sim) + '_base2.txt',
                                                                use_divergence=False,
                                                                div_method='js_div'
                                                                )

                # row = [[sim] + [internal_1[0]] + [internal_1[1]] + [internal_1[2]]]
                row = [[sim] + [internal_1[0]] + [internal_1[1]] + [internal_1[2]]]

                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerows(row)


def evaluate_all_sims(filename, alg='K-Means'):
    if alg == 'K-Means':
        header_labeled_data = ['Sim Nr', 'Nr features', 'ARI-a', 'AMI-a', 'ARI-a', 'AMI-a', 'ARI-a', 'AMI-a', 'ARI-a',
                               'AMI-a']
    else:
        header_labeled_data = ['Sim Nr', 'Nr features', 'ARI-a', 'AMI-a', 'ARI-nnp', 'AMI-nnp', 'ARI-a', 'AMI-a',
                               'ARI-nnp',
                               'AMI-nnp', 'ARI-a', 'AMI-a', 'ARI-nnp', 'AMI-nnp', 'ARI-a', 'AMI-a', 'ARI-nnp',
                               'AMI-nnp']

    rows = [header_labeled_data]

    sims = [4, 8, 14, 21, 22, 24, 29, 30, 33, 39, 46, 53, 54, 57, 59, 62, 64, 76, 77, 83, 84, 87, 89, 94]

    for i in sims:
        metrics_all = []
        row = [i, 79]

        metrics = do_spike_sorting(i, feature_extraction_method='pca', nb_components=2,
                                   weighted_method=True,
                                   nr_features=79, do_plot=False, do_save=False,
                                   folder='validation/sim' + str(i), alg='K-Means',
                                   scaling=False,
                                   minmax=True,
                                   divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                       i) + '_base2.txt',
                                   use_divergence=False,
                                   div_method='js_div')

        metrics = do_spike_sorting(i, feature_extraction_method='pca', nb_components=2,
                                   weighted_method=False,
                                   nr_features=79, do_plot=False, do_save=False,
                                   folder='validation/sim' + str(i), alg='K-Means',
                                   scaling=False,
                                   minmax=True,
                                   divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                       i) + '_base2.txt',
                                   use_divergence=False,
                                   div_method='js_div')

        for j in metrics:
            metrics_all.append(j)
            row.append(j)
            rows.append(row)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(rows)
