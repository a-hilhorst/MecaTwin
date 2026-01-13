"""
db_run_analysis.py

Provides all the analyses and plotting functions used to create the figures appearing in the publication:
"On the relationship between the stacking fault energy of FCC metallic alloys and their mechanical properties" by
Victor Trinquet, Gian-Marco Rignanese, Pascal J. Jacques, and Antoine Hilhorst
"""

# ============================
# Imports
# ============================
import operator
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib.transforms import Affine2D
import seaborn as sb

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors

from src.db_core_analysis import ELEMENT_LIST, CLUSTER_COLORS, MECHANISM_COLORS
from src.db_core_analysis import SAX_LABELS, CLUSTER11_LABELS, MECHANISM_LABELS
from src.db_core_analysis import create_df_from_feature_set, substract_series, moving_averaged_array
from src.db_core_analysis import b_p, kmeans_clustering_comp, df_to_latex_stats


# ============================
# Utility functions for plotting
# ============================
def scale_markers(_series, bounds, between_q1q3=True):
    scaled = bounds[0] + (_series - _series.min()) * (bounds[1] - bounds[0]) / (_series.max() - _series.min())

    if between_q1q3:
        q1 = scaled.quantile(0.25)
        q3 = scaled.quantile(0.75)

        scaled[(scaled < q1) | (scaled > q3)] = np.nan

        scaled = bounds[0] + (scaled - q1) * (bounds[1] - bounds[0]) / (q3 - q1)
    return scaled


def curate_df(_df, list_of_elements_to_drop=None):
    if list_of_elements_to_drop is None:
        list_of_elements_to_drop = ["Ti", "Pd", "V"]
    df_curated = _df.copy()
    list_proportion = []
    list_proportion_nb = []
    for c in df_curated.columns:
        df_tmp = df_curated[c]
        df_tmp = df_tmp.dropna()
        list_proportion_nb.append(len(df_tmp))
        list_proportion.append(len(df_tmp) / len(df_curated) * 100)
    df_prop = pd.DataFrame(index=df_curated.columns, data={"prop": list_proportion, "prop_nb": list_proportion_nb})
    df_prop = df_prop.sort_values(by="prop", ascending=False)

    # Sort the columns according to prop
    df_curated = df_curated.filter(df_prop.index, axis=1)

    df_curated = df_curated.drop(list_of_elements_to_drop, axis=1, errors="ignore")
    df_prop = df_prop.drop(list_of_elements_to_drop, axis=0, errors="ignore")

    # Values to display above each violin
    values = df_prop['prop'].to_numpy()
    values_nb = df_prop['prop_nb'].to_numpy()

    return df_curated, df_prop, values, values_nb


def violin_of_db(dfs, list_of_elements_to_drop=None, group_labels=None, side=None, save_fig=False):
    if list_of_elements_to_drop is None:
        list_of_elements_to_drop = ["Ti", "Pd", "V"]
    if group_labels is None:
        group_labels = [str(gl) for gl in np.arange(0, len(dfs))]

    l_df_violin = []
    l_df_prop = []
    l_values = []
    l_values_nb = []
    if isinstance(dfs, list):
        for dfi in dfs:
            df_violin_tmp, df_prop_tmp, values_tmp, values_nb_tmp = (
                curate_df(dfi, list_of_elements_to_drop=list_of_elements_to_drop))
            l_df_violin.append(df_violin_tmp)
            l_df_prop.append(df_prop_tmp)
            l_values.append(values_tmp)
            l_values_nb.append(values_nb_tmp)
    else:
        df_violin_tmp, df_prop_tmp, values_tmp, values_nb_tmp = (
            curate_df(dfs, list_of_elements_to_drop=list_of_elements_to_drop))
        l_df_violin.append(df_violin_tmp)
        l_df_prop.append(df_prop_tmp)
        l_values.append(values_tmp)
        l_values_nb.append(values_nb_tmp)

    # Create the violin plot
    _fig = go.Figure()

    for n, (df_violin, df_prop, values, values_nb, gl) in (
            enumerate(zip(l_df_violin, l_df_prop, l_values, l_values_nb, group_labels))):
        # Normalize the values to the [0, 1] range for color mapping
        normalized_values = (values - min(values)) / (max(values) - min(values))

        # Create a color scale using viridis
        colors = px.colors.sample_colorscale("viridis", normalized_values)

        fsize = 32
        for i, c in enumerate(df_violin.columns):
            # print(df_violin.columns)
            if side is None:
                if n % 2 == 0:
                    side = 'negative'
                else:
                    side = 'positive'
            _fig.add_trace(go.Violin(
                y=df_violin[c],
                box_visible=True,
                line_color='black',
                meanline_visible=False,
                fillcolor=colors[i],
                opacity=0.9,
                x0=c,
                side=side,
                span=[0, 1],
                width=1.75,
                points=False
            ))
            if n == 0:
                # Add the number above each violin
                _fig.add_annotation(
                    x=c,
                    y=1.1,  # Adjust the y position as needed
                    text=f"{values[i]:.1f}",
                    showarrow=False,
                    font=dict(size=fsize)
                )
                _fig.add_annotation(
                    x=c,
                    y=1.2,  # Adjust the y position as needed
                    text=f"{values_nb[i]}",
                    showarrow=False,
                    font=dict(size=fsize)
                )
    _fig.add_annotation(
        x=-1.025,
        y=1.1,  # Adjust the y position as needed
        text='%',
        showarrow=False,
        font=dict(size=fsize)
    )
    _fig.add_annotation(
        x=-1.025,
        y=1.2,  # Adjust the y position as needed
        text='#',
        showarrow=False,
        font=dict(size=fsize)
    )
    _fig.add_annotation(
        x=-0.075,
        y=0.925,  # Adjust the y position as needed
        xref='paper',
        yref='paper',
        text=' [at.%]',
        showarrow=False,
        font=dict(size=fsize)
    )

    # Update the layout
    _fig.update_layout(
        width=1250,
        height=900,
        showlegend=False,
        yaxis_zeroline=False,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=fsize),
            ticks="outside"
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=fsize),
            ticks="outside",
            tickmode='array',
            tickvals=[0, .25, .50, .75, 1],
            ticktext=['0', '25', '50', '75', '100']
        ),
        # title=dict(
        #     text='Violin Plot of Alloy Compositions',
        #     font=dict(size=20)
        # )
    )
    if isinstance(dfs, list):
        _fig.update_layout(violinmode='group', violingroupgap=0)

    if save_fig:
        _fig.write_image("violin_plot.png", scale=2)
    _fig.show()


def heatmap_of_db(
        _df, df_for_cluster=None, n_clusters=None, list_of_elements_to_drop=None, sort_by_list=None, scale=True):
    if list_of_elements_to_drop is None:
        list_of_elements_to_drop = ["Ti", "Pd", "V"]
    if sort_by_list is None:
        sort_by_list = ["Fe", "Co", "Ni", "Mn", "Cr", "Cu"]

    fsize = 32

    df_heatmap, _, _, _ = curate_df(_df, list_of_elements_to_drop=list_of_elements_to_drop)

    df_heatmap.fillna(0, inplace=True)

    if df_for_cluster is not None:
        df_for_cluster, _, _, _ = curate_df(df_for_cluster, list_of_elements_to_drop=list_of_elements_to_drop)
        df_for_cluster.fillna(0, inplace=True)

        df_for_cluster = kmeans_clustering_comp(df_for_cluster, n_clusters=n_clusters, scale=scale,
                                                list_of_elements=df_for_cluster.columns)
        df_heatmap['cluster'] = df_for_cluster.loc[df_heatmap.index, 'cluster']

        df_heatmap = df_heatmap[[*df_for_cluster.columns]]
    else:
        df_heatmap = kmeans_clustering_comp(df_heatmap, n_clusters=n_clusters, scale=scale,
                                            list_of_elements=df_heatmap.columns)

    df_heatmap = df_heatmap.sort_values(by=['cluster', *sort_by_list], ascending=False)
    clusters = df_heatmap['cluster']

    agg_dict = {col: ['mean', 'std', 'count'] for col in df_heatmap.select_dtypes(include='number') if col != 'cluster'}
    stats = df_heatmap.groupby('cluster').agg(agg_dict)

    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    for col in stats.columns:
        if col.endswith('_mean') or col.endswith('_std'):
            # stats[col] = stats[col].apply(lambda x: f"{x:.2f}")
            stats[col] = stats[col].apply(lambda x: f"{100 * x:.0f}")

    mean_std_cols = []
    for col in df_heatmap.select_dtypes(include='number'):
        if col == 'cluster':
            continue
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        new_col = f"{col}"
        stats[new_col] = stats[mean_col] + " ± " + stats[std_col]
        mean_std_cols.append(new_col)

    first_numeric = next(c for c in df_heatmap.select_dtypes(include='number') if c != 'cluster')
    count_col = f"{first_numeric}_count"

    final_stats = stats[[count_col] + mean_std_cols].rename(columns={count_col: 'count'})
    # latex_table = final_stats.to_latex(index=True, escape=False)
    # print(latex_table)

    final_stats_T = final_stats.T  # transpose
    final_stats_T.index.name = "Element"
    latex_table = final_stats_T.to_latex(index=True, escape=False)
    print(latex_table)

    df_heatmap = df_heatmap.drop(['cluster'], axis=1, errors="ignore")

    df_heatmap = df_heatmap.replace(0, float('nan'))
    df_heatmap = 100 * df_heatmap

    _fig = go.Figure(data=go.Heatmap(
        # x=df.T.columns,
        y=df_heatmap.T.index,
        z=df_heatmap.T.to_numpy(),
        # colorscale='tempo',
        # colorscale='matter',
        colorscale='Darkmint',
        colorbar=dict(
            title='at.%',
            # titlefont=dict(size=fsize),
            tickfont=dict(size=fsize),
            # titleside='right',
        ),
    ))

    # Update the layout
    _fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(
            tickfont=dict(size=fsize),
            tickmode='linear',
            tick0=0,
            dtick=100,
            ticks="outside",
            showgrid=False),
        yaxis=dict(
            tickfont=dict(size=fsize),
            autorange="reversed",
            ticks="outside",
            showgrid=False)
    )

    '''Adding cluster info'''
    unique_clusters = []
    for c in clusters:
        if len(unique_clusters) == 0 or unique_clusters[-1] != c:
            unique_clusters.append(c)

    annotations_x = []
    annotations_text = []

    start_idx = 0
    for uc in unique_clusters:
        block_len = np.sum(clusters == uc)
        middle_pos = start_idx + (block_len - 1) / 2

        if uc % 2 == 0:
            y_pos = -1
            shape_fill_color = 'lightgray'
        else:
            y_pos = -1.5
            shape_fill_color = 'white'

        # --- Add shaded rectangle above heatmap ---
        _fig.add_shape(
            type="rect",
            x0=start_idx - 0.5,  # start column
            x1=start_idx + block_len - 0.5,  # end column
            y0=-2.0,  # just above top row
            y1=-0.5,  # adjust thickness
            # fillcolor=shape_fill_color,
            fillcolor=CLUSTER_COLORS[uc],
            opacity=0.9,
            line_width=0,
            layer="below"  # keeps it behind the heatmap
        )

        # --- Add cluster label ---
        _fig.add_annotation(
            x=middle_pos,
            y=y_pos,  # place text in middle of shaded band
            text=str(uc),
            showarrow=False,
            # font=dict(size=fsize, color=CLUSTER_COLORS[uc]),
            font=dict(size=fsize, color='black'),
            xanchor="center",
            yanchor="middle"
        )

        start_idx += block_len

    _fig.show()

    return clusters


def label_mech(df_row, op=('gt', 0)):
    op_map = {
        'eq': operator.eq,  # equal to              (a == b)
        'ne': operator.ne,  # not equal to          (a != b)
        'lt': operator.lt,  # less than             (a < b)
        'le': operator.le,  # less or equal to      (a <= b)
        'gt': operator.gt,  # greater than          (a > b)
        'ge': operator.ge,  # greater or equal to   (a >= b)
    }

    label = ''
    if op_map[op[0]](df_row['is_twip'], op[1]):
        label += 'TWIP/'
    if op_map[op[0]](df_row['is_trip_epsilon'], op[1]):
        label += 'TRIPe/'
    if op_map[op[0]](df_row['is_trip_alpha'], op[1]):
        label += 'TRIPa/'
    if len(label) > 0:
        return label[:-1]
    else:
        return 'Other'


def plot_property_boxplots(_properties, _df_prop):
    plot_legend = True
    _fig, _sax = plt.subplots(2, int(len(_properties) / 2), figsize=(int(8 * len(_properties) / 4), 8))
    for nsax, (_property, _saxi, sax_label) in enumerate(zip(_properties, _sax.ravel(), SAX_LABELS)):
        if nsax > 0:
            plot_legend = False
        sb.boxplot(hue='Mechanism', y=_property[0], data=_df_prop, palette='Set2',
                   hue_order=MECHANISM_LABELS, showfliers=False,
                   ax=_saxi, legend=plot_legend)

        _ylims = _saxi.get_ylim()
        sb.stripplot(hue='Mechanism', y=_property[0], data=_df_prop, palette='Set2', alpha=0.6,
                     hue_order=MECHANISM_LABELS, jitter=True,
                     dodge=True, ax=_saxi, legend=False)
        _saxi.set(ylim=_ylims)

        _saxi.set(ylabel=None)
        _label = sax_label + _property[1]
        _saxi.text(.01, .99, _label, ha='left', va='top', transform=_saxi.transAxes)
        _saxi.grid(True, axis='y')

    first_sax = _sax.ravel()[0]
    _handles, _labels = first_sax.get_legend_handles_labels()
    first_sax.legend_.remove()
    _fig.legend(
        _handles, _labels,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.025)  # position below all subplots
    )

    # plt.tight_layout()
    plt.show()


def plot_averaged_mech_prop(x_curves, y_curves, method='mean', nx=100):
    x_max = 0
    for xc in x_curves:
        x_max = max(x_max, np.max(xc))
    x_common = np.linspace(0, x_max, num=nx)

    y_interp_list = []
    for xc, yc in zip(x_curves, y_curves):
        y_interp = interp1d(xc, yc, bounds_error=False, fill_value=np.nan)
        y_interp_list.append(y_interp(x_common))

    x_all = np.concatenate([x_common[~np.isnan(y)] for y in y_interp_list])
    y_all = np.concatenate([y[~np.isnan(y)] for y in y_interp_list])

    if method == 'mean':
        x_pred = x_common
        y_stack = np.vstack(y_interp_list)
        y_pred = np.nanmean(y_stack, axis=0)
        y_std = np.nanstd(y_stack, axis=0)
    else:
        x_pred = x_common
        y_pred = x_pred
        y_std = np.zeros(x_pred.shape)

    return x_pred, y_pred, y_std


def plot_sfe_tem_vs_tc_and_xrd():
    comp_features = [f'composition_at.{e}' for e in ELEMENT_LIST]

    # get df with clusters
    features = [
        'doc_id',
        *comp_features
    ]
    conditions = [{'feature': 'composition_at', 'op': 'exists'}]
    df_full = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_full.set_index("doc_id", inplace=True)

    features = [
        'doc_id',
        'thermodynamic_param.iSFE_GPR',
        'thermodynamic_param.iSFE_GPR_std',
        'thermodynamic_param.iSFE',
        'thermodynamic_param.SFE_xrd',
        'thermodynamic_param.iSFE_TC'
    ]
    conditions = [
        {'feature': 'thermodynamic_param.iSFE_GPR', 'op': 'ge', 'value': -1000},
        {'feature': 'thermodynamic_param.iSFE_TC', 'op': 'ge', 'value': -1000}
    ]
    df_sfe = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_sfe.set_index("doc_id", inplace=True)

    df_for_cluster, _, _, _ = curate_df(df_full, list_of_elements_to_drop=["Ti", "Pd", "V"])
    df_for_cluster.fillna(0, inplace=True)

    df_for_cluster = kmeans_clustering_comp(df_for_cluster, n_clusters=11, scale=False,
                                            list_of_elements=df_for_cluster.columns)

    df_sfe['cluster'] = df_for_cluster.loc[df_sfe.index, 'cluster']

    plt.rcParams.update({'font.size': 16})

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 10), layout='constrained')
    ax1.plot([-200, 200], [-200, 200], 'k--', linewidth=2)
    ax0.plot([-200, 200], [-200, 200], 'k--', linewidth=2)
    for cluster in CLUSTER11_LABELS.keys():
        df_sfe_cluster = df_sfe[df_sfe['cluster'] == cluster]
        if not df_sfe_cluster.empty:
            isfe_is_nan = df_sfe_cluster['iSFE'].isna()
            ax1.scatter(df_sfe_cluster['iSFE'][~isfe_is_nan], df_sfe_cluster['iSFE_TC'][~isfe_is_nan],
                        marker='x', linewidths=2, color=CLUSTER_COLORS[cluster], s=60, alpha=.9)
            ax1.scatter(df_sfe_cluster['iSFE_GPR'][isfe_is_nan], df_sfe_cluster['iSFE_TC'][isfe_is_nan],
                        marker='s', facecolors='none', edgecolors=CLUSTER_COLORS[cluster], s=60, linewidths=2,
                        alpha=.9, label=CLUSTER11_LABELS[cluster])

            ax0.scatter(df_sfe_cluster['iSFE'][~isfe_is_nan], df_sfe_cluster['SFE_xrd'][~isfe_is_nan],
                        marker='x', linewidths=2, color=CLUSTER_COLORS[cluster], s=60, alpha=.9)
            ax0.scatter(df_sfe_cluster['iSFE_GPR'][isfe_is_nan], df_sfe_cluster['SFE_xrd'][isfe_is_nan],
                        marker='s', facecolors='none', edgecolors=CLUSTER_COLORS[cluster], s=60, linewidths=2,
                        alpha=.9)

    ax1.set_aspect('equal')
    ax1.set(ylim=(-100, 100), xlim=(-0, 80),
            xlabel='SFE (measured or modeled) [mJ/m2]', ylabel='SFE (thermodynamic prediction) [mJ/m2]')
    fig.legend(frameon=False, loc='outside lower left', ncols=4, mode="expand", borderaxespad=0.)
    ax0.set(ylim=(-0, 100), xlim=(-0, 100),
            xlabel='SFE (measured by TEM or modeled) [mJ/m2]', ylabel='SFE (measured by diffraction) [mJ/m2]')

    ax1.text(0.01, 0.99, " b)",
             transform=ax1.transAxes,
             ha='left', va='top',
             fontsize=22, color='black')
    ax0.text(0.01, 0.99, " a)",
             transform=ax0.transAxes,
             ha='left', va='top',
             fontsize=22, color='black')

    plt.show()


# ============================
# Analysis of the datasets
# ============================
def main(plot_venn_diagram=False,
         plot_violin=False,
         plot_heatmap=False,
         plot_sfe_vs_lit_alloying_elt=False,
         box_plots_of_properties=False,
         average_tensile_curves_based_on_cluster=False,
         dsa_vs_no_dsa=False,
         plots_average_hardening=False,
         plots_average_hardening_clustered=False,
         plot_sfe_tem_vs_tc_vs_xrd=False):
    """
    Creates the figures appearing in the publication:
    "On the relationship between the stacking fault energy of FCC metallic alloys and their mechanical properties" by
    Victor Trinquet, Gian-Marco Rignanese, Pascal J. Jacques, and Antoine Hilhorst

    """

    '''Description of the database'''
    # Generate Dataframe for plotting
    comp_features = [f'composition_at.{e}' for e in ELEMENT_LIST]

    features = [
        'doc_id',
        *comp_features
    ]
    conditions = [{'feature': 'composition_at', 'op': 'exists'}]
    df_full = (create_df_from_feature_set({
        'features': [*features, 'thermodynamic_param.iSFE', 'thermodynamic_param.iSFE_GPR'], 'conditions': conditions},
        source="myDatabase_ext.json"))

    conditions = [{'feature': 'thermodynamic_param.iSFE', 'op': 'ge', 'value': -1000}]
    df_isfe = (create_df_from_feature_set({'features': features, 'conditions': conditions},
                                          source="myDatabase_ext.json"))

    conditions = [{'feature': 'mechanical_param.true_ultimate_stress', 'op': 'ge', 'value': 0}]
    df_mech = (create_df_from_feature_set({'features': features, 'conditions': conditions},
                                          source="myDatabase_ext.json"))

    conditions = [{'feature': 'thermodynamic_param.SFE_xrd', 'op': 'ge', 'value': -1000}]
    df_isfe_xrd = (create_df_from_feature_set({'features': features, 'conditions': conditions},
                                              source="myDatabase_ext.json"))

    df_full.set_index("doc_id", inplace=True)
    df_isfe.set_index("doc_id", inplace=True)
    df_mech.set_index("doc_id", inplace=True)
    df_isfe_xrd.set_index("doc_id", inplace=True)

    df_full = df_full.drop(['iSFE', 'iSFE_GPR'], axis=1, errors="ignore")

    # Plot Venn Diagrams
    if plot_venn_diagram:
        do_rotation = False
        viridis = plt.get_cmap('viridis', 3)
        colors = [viridis(i) for i in range(3)]

        fig, ax = plt.subplots(figsize=(12, 12))
        v = venn3((set(df_isfe.index), set(df_mech.index), set(df_isfe_xrd.index)),
                  ('SFE (TEM)', 'Mech.', 'SFE (XRD)'), set_colors=colors, ax=ax)

        for label in v.set_labels:
            if label:
                label.set_fontsize(24)
        for label in v.subset_labels:
            if label:
                label.set_fontsize(20)

        if do_rotation:
            rotation = Affine2D().rotate_deg(90) + ax.transData
            for patch in v.patches:
                if patch:  # some patches may be None
                    patch.set_transform(rotation)
            for label in v.set_labels:
                if label:
                    label.set_transform(rotation)
                    if label.get_text() == 'SFE (XRD)':
                        (x, _) = label.get_position()
                        label.set_x(x+0.075)
            for label in v.subset_labels:
                if label:
                    label.set_transform(rotation)

            ax.set_aspect('equal')
            ax.autoscale_view()  # Recalculate limits
            ax.margins(0.2)

        plt.show()

    # Violin plots
    if plot_violin:
        violin_of_db(df_full, save_fig=True)
        violin_of_db(df_isfe, side='negative')
        violin_of_db(df_mech, side='positive')

    # Heatmaps
    if plot_heatmap:
        heatmap_of_db(df_full, n_clusters=11, scale=False)

    # Effect of alloying element on SFE
    if plot_sfe_vs_lit_alloying_elt:
        alloys = [
            {'Fe': 0.7, 'Mn': 0.3},
            {'Fe': 0.74, 'Cr': 0.18, 'Ni': 0.08},
            {'Fe': 0.4, 'Mn': 0.4, 'Co': 0.1, 'Cr': 0.1},
            {'Fe': 0.4, 'Co': 0.2, 'Cr': 0.2, 'Ni': 0.2}
        ]
        elt_additions = [
            ['Cr', 'Ni', 'Mn', 'Si', 'C'],
            ['Cr', 'Ni', 'Mn', 'Si', 'C'],
            ['Cr', 'Ni', 'Mn', 'Si', 'C'],
            ['Cr', 'Ni', 'Mn', 'Si', 'C'],
        ]

        colormap = plt.get_cmap('tab10', 10)
        elt_colors = [colormap(i) for i in range(10)]
        colors = {
            'Cr': elt_colors[0],
            'Ni': elt_colors[1],
            'Mn': elt_colors[2],
            'Si': elt_colors[3],
            'C': elt_colors[4],
            'Al': elt_colors[5],
            'Co': elt_colors[6],
        }

        plt.rcParams.update({'font.size': 16})

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
        axs = [ax0, ax1, ax2, ax3]

        ax0.set(ylabel='Predicted SFE [mJ/m2]')
        ax2.set(xlabel='Alloying addition, X [at.%]', ylabel='Predicted SFE [mJ/m2]')
        ax3.set(xlabel='Alloying addition, X [at.%]')

        for elt in elt_additions[-1]:
            ax2.plot(np.nan, np.nan, '-', color=colors[elt], label=elt)
        ax2.legend(loc='lower right')

        text_labels = [
            'a) (Fe$_{70}$Mn$_{30}$)$_{100-y}$X$_y$',
            'b) (Fe$_{74}$Cr$_{18}$Ni$_{8}$)$_{100-y}$X$_y$',
            'c) (Fe$_{40}$Mn$_{40}$Co$_{10}$Cr$_{10}$)$_{100-y}$X$_y$',
            'd) (Fe$_{40}$Co$_{20}$Cr$_{20}$Ni$_{20}$)$_{100-y}$X$_y$'
        ]
        for tl, ax in zip(text_labels, axs):
            ax.text(0.015, 0.985, tl, ha='left', va='top', transform=ax.transAxes)

        fig.tight_layout()
        plt.show()

    # Generate db for plotting
    comp_features = [f'composition_at.{e}' for e in ELEMENT_LIST]

    features = [
        'doc_id',
        *comp_features,
        'mechanical_param.is_twip',
        'mechanical_param.is_trip_epsilon',
        'mechanical_param.is_trip_alpha',
        'mechanical_param.true_stress',
        'mechanical_param.true_strain',
        'mechanical_param.serrated_yielding',
        'mechanical_param.true_ultimate_stress',
        'mechanical_param.true_uniform_strain',
        'mechanical_param.yield_strength',
        'mechanical_param.nincr_mean',
        'mechanical_param.nincr_max',
        'mechanical_param.plastic_work',
        'mechanical_param.grain_size',
        'matminer_features.Yang delta',
        'matminer_features.Yang omega',
        'matminer_features.Mean cohesive energy',
        'matminer_features.Electronegativity delta',
        'matminer_features.Radii local mismatch',
        'matminer_features.Shear modulus mean',
        'thermodynamic_param.iSFE_GPR',
        'thermodynamic_param.gFCC2BCC',
        'thermodynamic_param.gFCC2HCP',
        'thermodynamic_param.a_fcc_pred',
        'thermodynamic_param.a_hcp_pred'
    ]
    conditions = [{'feature': 'mechanical_param.true_ultimate_stress', 'op': 'ge', 'value': 0}]
    df_prop = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_prop.set_index("doc_id", inplace=True)

    features = [
        'doc_id',
        *comp_features
    ]
    conditions = [{'feature': 'composition_at', 'op': 'exists'}]
    df_full = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_full.set_index("doc_id", inplace=True)

    df_for_cluster, _, _, _ = curate_df(df_full, list_of_elements_to_drop=["Ti", "Pd", "V"])
    df_for_cluster.fillna(0, inplace=True)

    df_for_cluster = kmeans_clustering_comp(df_for_cluster, n_clusters=11, scale=False,
                                            list_of_elements=df_for_cluster.columns)

    # Add clusters and mechanisms
    df_prop['cluster'] = df_for_cluster.loc[df_prop.index, 'cluster']
    df_prop['Mechanism'] = df_prop.apply((lambda _x: label_mech(_x, op=('gt', 0))), axis=1)

    # Fill nan
    df_prop.fillna(0, inplace=True)

    # Add parameter(s) based on composition
    df_prop['CN_content'] = df_prop['C'] + df_prop['N']

    # Add parameter(s) based on thermodynamic properties
    df_prop['a_fcc_pred/a_hcp_pred'] = df_prop['a_fcc_pred'] / df_prop['a_hcp_pred']
    df_prop['b_p'] = df_prop.apply(b_p, axis=1)
    df_prop['iSFE/Gb'] = 10 * df_prop['iSFE_GPR'] / df_prop['Shear modulus mean'] / df_prop['b_p']

    # Add parameter(s) based on mechanical properties
    df_prop['UTS-YS'] = df_prop['true_ultimate_stress'] - df_prop['yield_strength']
    df_prop['PW-YS*EU'] = df_prop['plastic_work'] - df_prop['yield_strength'] * df_prop['true_uniform_strain']
    df_prop['PW/EU-YS'] = df_prop['plastic_work'] / df_prop['true_uniform_strain'] - df_prop['yield_strength']
    df_prop['f_CRSStw1'] = 10 * df_prop['iSFE_GPR'] / df_prop['b_p'] / df_prop['true_ultimate_stress']
    df_prop['f_CRSStw2'] = 10 * df_prop['iSFE_GPR'] / df_prop['b_p'] / df_prop['yield_strength']

    '''Plotting Mechanical Properties'''
    # Generate Dataframe
    comp_features = [f'composition_at.{e}' for e in ELEMENT_LIST]

    features = [
        'doc_id',
        *comp_features,
        'mechanical_param.is_twip',
        'mechanical_param.is_trip_epsilon',
        'mechanical_param.is_trip_alpha',
        'mechanical_param.true_stress',
        'mechanical_param.true_strain',
        'mechanical_param.serrated_yielding',
        'mechanical_param.true_ultimate_stress',
        'mechanical_param.true_uniform_strain',
        'mechanical_param.yield_strength',
        'mechanical_param.nincr_mean',
        'mechanical_param.nincr_max',
        'mechanical_param.plastic_work',
        'mechanical_param.grain_size',
        'matminer_features.Yang delta',
        'matminer_features.Yang omega',
        'matminer_features.Mean cohesive energy',
        'matminer_features.Electronegativity delta',
        'matminer_features.Radii local mismatch',
        'matminer_features.Shear modulus mean',
        'thermodynamic_param.iSFE_GPR',
        'thermodynamic_param.gFCC2BCC',
        'thermodynamic_param.gFCC2HCP',
        'thermodynamic_param.a_fcc_pred',
        'thermodynamic_param.a_hcp_pred'
    ]
    conditions = [{'feature': 'mechanical_param.true_ultimate_stress', 'op': 'ge', 'value': 0}]
    df_prop = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_prop.set_index("doc_id", inplace=True)

    features = [
        'doc_id',
        *comp_features
    ]
    conditions = [{'feature': 'composition_at', 'op': 'exists'}]
    df_full = create_df_from_feature_set({'features': features, 'conditions': conditions})
    df_full.set_index("doc_id", inplace=True)

    df_for_cluster, _, _, _ = curate_df(df_full, list_of_elements_to_drop=["Ti", "Pd", "V"])
    df_for_cluster.fillna(0, inplace=True)

    df_for_cluster = kmeans_clustering_comp(df_for_cluster, n_clusters=11, scale=False,
                                            list_of_elements=df_for_cluster.columns)

    # Add clusters and mechanisms
    df_prop['cluster'] = df_for_cluster.loc[df_prop.index, 'cluster']
    df_prop['Mechanism'] = df_prop.apply((lambda _x: label_mech(_x, op=('gt', 0))), axis=1)

    # Fill nan
    df_prop.fillna(0, inplace=True)

    # Add parameter(s) based on composition
    df_prop['CN_content'] = df_prop['C'] + df_prop['N']

    # Add parameter(s) based on thermodynamic properties
    df_prop['a_fcc_pred/a_hcp_pred'] = df_prop['a_fcc_pred'] / df_prop['a_hcp_pred']
    df_prop['b_p'] = df_prop.apply(b_p, axis=1)
    df_prop['iSFE/Gb'] = 10 * df_prop['iSFE_GPR'] / df_prop['Shear modulus mean'] / df_prop['b_p']

    # Add parameter(s) based on mechanical properties
    df_prop['UTS-YS'] = df_prop['true_ultimate_stress'] - df_prop['yield_strength']
    df_prop['PW-YS*EU'] = df_prop['plastic_work'] - df_prop['yield_strength'] * df_prop['true_uniform_strain']
    df_prop['PW/EU-YS'] = df_prop['plastic_work'] / df_prop['true_uniform_strain'] - df_prop['yield_strength']
    df_prop['f_CRSStw1'] = 10 * df_prop['iSFE_GPR'] / df_prop['b_p'] / df_prop['true_ultimate_stress']
    df_prop['f_CRSStw2'] = 10 * df_prop['iSFE_GPR'] / df_prop['b_p'] / df_prop['yield_strength']

    # Box plots of properties
    if box_plots_of_properties:
        properties = [
            ('true_ultimate_stress', 'UTS [MPa]'),
            ('true_uniform_strain', 'EU [-]'),
            ('yield_strength', 'YS [MPa]'),
            ('nincr_mean', 'n [-]'),
            ('plastic_work', 'Plastic work [MPa]'),
            ('iSFE/Gb', 'SFE/Gb [mJ/m2]'),
            ('gFCC2BCC', 'Gfcc - Gbcc [J]'),
            ('gFCC2HCP', 'Gfcc - Ghcp [J]')
        ]
        plot_property_boxplots(properties, df_prop)

        properties_set1 = [
            ('true_ultimate_stress', 'UTS [MPa]'),
            ('true_uniform_strain', 'EU [-]'),
            ('yield_strength', 'YS [MPa]'),
            ('nincr_mean', 'n [-]')
        ]
        plot_property_boxplots(properties_set1, df_prop)

        properties_set2 = [
            ('gFCC2BCC', 'Gfcc - Gbcc [J]'),
            ('gFCC2HCP', 'Gfcc - Ghcp [J]'),
            ('iSFE/Gb', 'SFE/Gb [mJ/m2]'),
            ('a_fcc_pred/a_hcp_pred', 'afcc / ahcp [-]')
        ]
        plot_property_boxplots(properties_set2, df_prop)

        elt_subset = ELEMENT_LIST.copy()
        elt_subset.remove('V')
        elt_subset.remove('Pd')
        elt_subset.remove('Zn')
        df_prop.fillna(0, inplace=True)
        table = df_to_latex_stats(
            df_prop,
            [*elt_subset, 'true_ultimate_stress', 'true_uniform_strain', 'yield_strength', 'nincr_mean'],
            'Mechanism', decimals=1
        )
        print(table)

    '''Average tensile curves based on cluster'''
    if average_tensile_curves_based_on_cluster:
        subfeatures = [
            'is_twip',
            'is_trip_epsilon',
            'is_trip_alpha',
            'true_stress',
            'true_strain',
            'yield_strength',
            'serrated_yielding',
            'Mechanism',
            'cluster'
        ]

        df_mech_prop = df_prop.copy()
        df_mech_prop = df_mech_prop[subfeatures]

        '''Taking into account serrated yielding'''
        if dsa_vs_no_dsa:
            print(df_mech_prop.groupby('serrated_yielding').size())

            fig, ax = plt.subplots(1, 2, figsize=(6, 6))
            for ci, cluster in enumerate(df_mech_prop['cluster'].unique()):
                df_dsa = df_mech_prop[(df_mech_prop['cluster'] == cluster) & (df_mech_prop['serrated_yielding'] == 1)]
                df_nodsa = df_mech_prop[(df_mech_prop['cluster'] == cluster) & (df_mech_prop['serrated_yielding'] == 0)]

                # DSA
                if not df_dsa.empty:
                    strains = df_dsa['true_strain']
                    stresses = df_dsa['true_stress']
                    ys = df_dsa['yield_strength']
                    x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, ys), method='mean')

                    ax[0].plot(x, y, color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])
                    ax[0].fill_between(x.flatten(), y - std, y + std, color=CLUSTER_COLORS[cluster], alpha=0.2)

                # no DSA
                if not df_nodsa.empty:
                    strains = df_nodsa['true_strain']
                    stresses = df_nodsa['true_stress']
                    ys = df_nodsa['yield_strength']
                    x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, ys), method='mean')

                    ax[1].plot(x, y, color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])
                    ax[1].fill_between(x.flatten(), y - std, y + std, color=CLUSTER_COLORS[cluster], alpha=0.2)

            ax[0].legend()
            ax[0].set(ylim=[0, 2500])

            ax[1].legend()
            ax[1].set(ylim=[0, 2500])

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            for ci, cluster in enumerate(df_mech_prop['cluster'].unique()):
                df_dsa = df_mech_prop[(df_mech_prop['cluster'] == cluster) & (df_mech_prop['serrated_yielding'] == 1)]
                df_nodsa = df_mech_prop[(df_mech_prop['cluster'] == cluster) & (df_mech_prop['serrated_yielding'] == 0)]

                # DSA
                if not df_dsa.empty:
                    strains = df_dsa['true_strain']
                    stresses = df_dsa['true_stress']
                    ys = df_dsa['yield_strength']
                    x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, ys), method='mean')

                    ax.plot(x, y, '--', color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])

                # no DSA
                if not df_nodsa.empty:
                    strains = df_nodsa['true_strain']
                    stresses = df_nodsa['true_stress']
                    ys = df_nodsa['yield_strength']
                    x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, ys), method='mean')

                    ax.plot(x, y, color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])
            ax.legend()
            plt.show()

        '''Averages with strain hardening (CLUSTERS)'''
        alpha_value = 0.1
        vs_stress = 1
        w_dsde_e_s = 0
        nx = 100
        wl1 = 11
        wl2 = 5
        wl3 = 11

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for cluster in CLUSTER11_LABELS.keys():
            df_mech_prop_cluster = df_mech_prop[df_mech_prop['cluster'] == cluster]
            if not df_mech_prop_cluster.empty:
                strains = df_mech_prop_cluster['true_strain']
                stresses = df_mech_prop_cluster['true_stress']
                yss = df_mech_prop_cluster['yield_strength']

                strain_hardenings = []
                stresses_ma = []
                strains_ma = []
                for strain, stress, ys in zip(strains, stresses, yss):
                    true_strain_filt = savgol_filter(strain, wl1, 3)
                    true_stress_filt = savgol_filter(stress, wl1, 3)

                    ds = np.diff(true_stress_filt)
                    de = np.diff(true_strain_filt)

                    true_strain_avg = moving_averaged_array(true_strain_filt)
                    true_stress_avg = moving_averaged_array(true_stress_filt)
                    dsde = np.divide(ds, de, out=np.zeros_like(ds), where=de != 0)
                    dsde_e_s = np.divide(np.multiply(dsde, true_strain_avg), true_stress_avg)

                    ind = (dsde > true_stress_avg) & (true_stress_avg > ys)
                    strains_ma.append(true_strain_avg[ind])
                    stresses_ma.append(true_stress_avg[ind])
                    if w_dsde_e_s:
                        strain_hardenings.append(savgol_filter(dsde_e_s[ind], wl2, 3))
                    else:
                        strain_hardenings.append(savgol_filter(dsde[ind], wl2, 3))

                x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, yss), method='mean', nx=nx)

                ax[0].plot(x, y, linewidth=2, color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])
                ax[0].fill_between(x.flatten(), y - std, y + std, color=CLUSTER_COLORS[cluster], alpha=alpha_value)

                if vs_stress:
                    x, y, std = plot_averaged_mech_prop(substract_series(stresses_ma, yss), strain_hardenings,
                                                        method='mean', nx=nx)
                else:
                    x, y, std = plot_averaged_mech_prop(strains_ma, strain_hardenings, method='mean', nx=nx)

                x = x[~np.isnan(y)]
                std = std[~np.isnan(y)]
                y = y[~np.isnan(y)]
                if wl3:
                    yfilt = savgol_filter(y, wl3, 3)
                    y = yfilt
                    stdfilt = savgol_filter(std, wl3, 3)
                    std = stdfilt

                ax[1].plot(x, y, linewidth=2, color=CLUSTER_COLORS[cluster], label=CLUSTER11_LABELS[cluster])
                ax[1].fill_between(x.flatten(), y - std, y + std, color=CLUSTER_COLORS[cluster], alpha=alpha_value)
        ax[0].legend()

        ax[0].set(xlim=[0, .8], ylim=[0, 2100], xlabel='True strain [-]',
                  ylabel='True stress - yield strength (avg.) [MPa]')
        if not vs_stress:
            ax[1].set(xlim=[0, .8], xlabel='True strain [-]')
        else:
            ax[1].set(xlim=[0, 2100], xlabel='True stress - yield strength (avg.) [MPa]')
        if w_dsde_e_s:
            ax[1].set(ylim=[0, 1.5], ylabel='Averaged strain hardening coefficient [-]')
        else:
            ax[1].set(ylim=[0, 5000], ylabel='Averaged strain hardening rate [MPa]')

        '''Averages with strain hardening (MECHANISMS)'''
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for mechanism in MECHANISM_LABELS:
            df_mech_prop_mech = df_mech_prop[df_mech_prop['Mechanism'] == mechanism]
            if not df_mech_prop_mech.empty:
                strains = df_mech_prop_mech['true_strain']
                stresses = df_mech_prop_mech['true_stress']
                yss = df_mech_prop_mech['yield_strength']

                strain_hardenings = []
                stresses_ma = []
                strains_ma = []
                for strain, stress, ys in zip(strains, stresses, yss):
                    true_strain_filt = savgol_filter(strain, wl1, 3)
                    true_stress_filt = savgol_filter(stress, wl1, 3)

                    ds = np.diff(true_stress_filt)
                    de = np.diff(true_strain_filt)

                    true_strain_avg = moving_averaged_array(true_strain_filt)
                    true_stress_avg = moving_averaged_array(true_stress_filt)
                    dsde = np.divide(ds, de, out=np.zeros_like(ds), where=de != 0)
                    dsde_e_s = np.divide(np.multiply(dsde, true_strain_avg), true_stress_avg)

                    ind = (dsde > true_stress_avg) & (true_stress_avg > ys)
                    strains_ma.append(true_strain_avg[ind])
                    stresses_ma.append(true_stress_avg[ind])
                    if w_dsde_e_s:
                        strain_hardenings.append(savgol_filter(dsde_e_s[ind], wl2, 3))
                    else:
                        strain_hardenings.append(savgol_filter(dsde[ind], wl2, 3))

                x, y, std = plot_averaged_mech_prop(strains, substract_series(stresses, yss), method='mean', nx=nx)

                ax[0].plot(x, y, linewidth=2, color=MECHANISM_COLORS[mechanism], label=mechanism)
                ax[0].fill_between(x.flatten(), y - std, y + std, color=MECHANISM_COLORS[mechanism], alpha=alpha_value)

                if vs_stress:
                    x, y, std = plot_averaged_mech_prop(substract_series(stresses_ma, yss), strain_hardenings,
                                                        method='mean', nx=nx)
                else:
                    x, y, std = plot_averaged_mech_prop(strains_ma, strain_hardenings, method='mean', nx=nx)

                x = x[~np.isnan(y)]
                std = std[~np.isnan(y)]
                y = y[~np.isnan(y)]
                if wl3:
                    yfilt = savgol_filter(y, wl3, 3)
                    y = yfilt
                    stdfilt = savgol_filter(std, wl3, 3)
                    std = stdfilt

                ax[1].plot(x, y, linewidth=2, color=MECHANISM_COLORS[mechanism], label=mechanism)
                ax[1].fill_between(x.flatten(), y - std, y + std, color=MECHANISM_COLORS[mechanism], alpha=alpha_value)
        ax[0].legend()
        # ax[1].legend()

        ax[0].set(xlim=[0, .8], ylim=[0, 2100], xlabel='True strain [-]',
                  ylabel='True stress - yield strength (avg.) [MPa]')
        if not vs_stress:
            ax[1].set(xlim=[0, .8], xlabel='True strain [-]')
        else:
            ax[1].set(xlim=[0, 2100], xlabel='True stress - yield strength (avg.) [MPa]')
        if w_dsde_e_s:
            ax[1].set(ylim=[0, 1.5], ylabel='Averaged strain hardening coefficient [-]')
        else:
            ax[1].set(ylim=[0, 5000], ylabel='Averaged strain hardening rate [MPa]')

        plt.show()

    '''Plots of average strain hardening vs different quantities'''
    if plots_average_hardening:
        fig, sax = plt.subplots(1, 2, figsize=(16, 8))

        ind_CN = df_prop['CN_content'] < 0.005
        sizes_wCN = scale_markers(df_prop['gFCC2HCP'], (10, 100))
        sizes_woCN = scale_markers(df_prop['Mean cohesive energy'], (10, 100))

        for mech in MECHANISM_LABELS:
            mech_ind = df_prop['Mechanism'] == mech

            ind_wCN_mech = mech_ind & ~ind_CN
            ind_woCN_mech = mech_ind & ind_CN

            sax[0].scatter(df_prop['iSFE/Gb'][ind_wCN_mech], df_prop['PW/EU-YS'][ind_wCN_mech],
                           s=sizes_wCN[ind_wCN_mech], c=MECHANISM_COLORS[mech], alpha=0.5)
            sax[1].scatter(df_prop['iSFE/Gb'][ind_woCN_mech], df_prop['PW/EU-YS'][ind_woCN_mech],
                           s=sizes_woCN[ind_woCN_mech], c=MECHANISM_COLORS[mech], alpha=0.5)

        sax[0].set(ylim=(200, 900), xlim=(10, 50), ylabel='PW/EU-YS', xlabel='SFE/Gb')
        sax[1].set(ylim=(200, 900), xlim=(10, 50), ylabel='PW/EU-YS', xlabel='SFE/Gb')

        plt.show()

    '''With clusters'''
    plt.rcParams.update({'font.size': 16})
    if plots_average_hardening_clustered:
        no_sy = df_prop['serrated_yielding'] < 0.5
        target, target_bounds = 'PW/EU-YS', (0, 800)

        draw_neighbours = False

        fig, sax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 8), constrained_layout=True)
        clabenum = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ']
        for cluster, (cle, cluster_color, saxi) in (
                enumerate(zip(clabenum, CLUSTER_COLORS, sax.ravel()))):
            cluster_ind = df_prop['cluster'] == cluster

            if draw_neighbours:
                df_neighbours = df_prop[ELEMENT_LIST][cluster_ind & no_sy]
                xx = df_prop['iSFE/Gb'][cluster_ind & no_sy]
                yy = df_prop[target][cluster_ind & no_sy]
                if not df_neighbours.empty:
                    df_neighbours_values = df_neighbours.values

                    n_neighbors = 3  # include self + 2 neighbors
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(df_neighbours_values)
                    _, indices = nbrs.kneighbors(df_neighbours_values)

                    for i, idx in enumerate(df_neighbours.index):
                        for j in indices[i][1:]:
                            x_i = xx.iloc[i]
                            y_i = yy.iloc[i]
                            x_j = xx.iloc[j]
                            y_j = yy.iloc[j]
                            saxi.plot([x_i, x_j], [y_i, y_j], color='gray', alpha=0.3, linewidth=1)

            if cluster == 1:
                fig_carbon, ax_carbon = plt.subplots(1, 2,
                                                     sharey=True, figsize=(10, 6), constrained_layout=True)
                for mech in MECHANISM_LABELS:
                    mech_ind = df_prop['Mechanism'] == mech
                    ind_carbon = cluster_ind & mech_ind & ~no_sy

                    target_intersticial = 'C'

                    if df_prop[target_intersticial][ind_carbon].size > 1:
                        sizes_carbon0 = scale_markers(df_prop[target_intersticial][ind_carbon], (10, 100),
                                                      between_q1q3=False)
                        sizes_carbon1 = scale_markers(df_prop['iSFE/Gb'][ind_carbon], (10, 100), between_q1q3=False)

                        ax_carbon[0].scatter(df_prop['iSFE/Gb'][ind_carbon], df_prop[target][ind_carbon],
                                             s=sizes_carbon0, c=MECHANISM_COLORS[mech], alpha=0.8)
                        ax_carbon[0].set(ylim=(100, 1000), xlim=(10, 30),
                                         ylabel='Avg. Strain Hardening [MPa]', xlabel=r'SFE/Gb [mJ/m$^{-2}$]')

                        ax_carbon[1].scatter(df_prop[target_intersticial][ind_carbon], df_prop[target][ind_carbon],
                                             s=sizes_carbon1, c=MECHANISM_COLORS[mech], alpha=0.8)
                        ax_carbon[1].set(ylim=(100, 1000), xlim=(0.01, 0.06),
                                         ylabel='Avg. Strain Hardening [MPa]', xlabel=r'C content [at.]')

                        ax_carbon[0].label_outer()
                        ax_carbon[1].label_outer()
                        ax_carbon[0].text(0.025, 0.975, 'a) ' + CLUSTER11_LABELS[1], ha='left', va='top',
                                          transform=ax_carbon[0].transAxes)
                        ax_carbon[1].text(0.025, 0.975, 'b) ' + CLUSTER11_LABELS[1], ha='left', va='top',
                                          transform=ax_carbon[1].transAxes)

                # legend
                mrks0 = [ax_carbon[0].scatter([], [], s=10, c='black', alpha=0.8, label=' '),
                         ax_carbon[0].scatter([], [], s=30, c='black', alpha=0.8, label=' '),
                         ax_carbon[0].scatter([], [], s=70, c='black', alpha=0.8, label=' ')]
                l1 = ax_carbon[0].legend(ncol=3, handles=mrks0, title=r'C content [at.]', loc='lower center',
                                         scatterpoints=1, columnspacing=0.25, frameon=False, fontsize=12)
                clrs0 = [ax_carbon[0].scatter([], [], s=70, c=MECHANISM_COLORS['TWIP'], alpha=0.8, label='TWIP'),
                         ax_carbon[0].scatter([], [], s=70, c=MECHANISM_COLORS['TWIP/TRIPe'],
                                              alpha=0.8, label='TWIP/TRIPe')]
                ax_carbon[0].legend(ncol=2, handles=clrs0, columnspacing=0.25, frameon=False,
                                    loc='center', bbox_to_anchor=(0.5, 1.05))
                ax_carbon[0].add_artist(l1)

                mrks1 = [ax_carbon[1].scatter([], [], s=10, c='black', alpha=0.8, label=' '),
                         ax_carbon[1].scatter([], [], s=30, c='black', alpha=0.8, label=' '),
                         ax_carbon[1].scatter([], [], s=70, c='black', alpha=0.8, label=' ')]
                ax_carbon[1].legend(ncol=3, handles=mrks1, title=r'SFE/Gb [mJ/m$^{-2}$]', loc='lower center',
                                    scatterpoints=1, columnspacing=0.25, frameon=False, fontsize=12)
                # fig_carbon.savefig(fname=f'avgstrhard_vs_sfe_and_c_femnc.png', format='png', dpi=300)

            for mech in MECHANISM_LABELS:
                mech_ind = df_prop['Mechanism'] == mech
                ind = cluster_ind & mech_ind & no_sy

                saxi.scatter(df_prop['iSFE/Gb'][ind], df_prop[target][ind],
                             s=50, c=MECHANISM_COLORS[mech], alpha=0.85)

            saxi.set(ylim=target_bounds, xlim=(10, 60),
                     ylabel='Avg. Strain Hardening [MPa]', xlabel=r'SFE/Gb [mJ/m$^{-2}$]')
            saxi.text(0.025, 0.975, cle + CLUSTER11_LABELS[cluster], ha='left', va='top', transform=saxi.transAxes)
            saxi.label_outer()

        # fig.savefig(fname=f'avgstrhard_vs_sfe_per_cluster.png', format='png', dpi=300)
        hdles = []
        for mech in MECHANISM_LABELS:
            hdle = sax[1][1].scatter([], [], s=50, c=MECHANISM_COLORS[mech], alpha=0.85, label=mech)
            hdles.append(hdle)
        sax[1][1].legend(handles=hdles, ncol=4, loc='upper center', framealpha=1)
        fig.savefig(fname=f'avgstrhard_vs_sfe_per_cluster_legend.png', format='png', dpi=300)
        plt.show()

        def subplots_target1_vs2_by_cluster_mech(
                _df, targetx, targety, _sizes, targetx_bounds, targety_bounds, targetx_label, _no_sy_ind):
            _, _sax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 8), constrained_layout=True)

            for _cluster, (_cluster_label, _cluster_color, _saxi) in enumerate(
                    zip(CLUSTER11_LABELS, CLUSTER_COLORS, _sax.ravel())):
                _cluster_ind = _df['cluster'] == _cluster
                for _mech in MECHANISM_LABELS:
                    _mech_ind = _df['Mechanism'] == _mech
                    _ind = _cluster_ind & _mech_ind & _no_sy_ind

                    _saxi.scatter(_df[targetx][_ind], _df[targety][_ind],
                                  s=_sizes[_ind], c=MECHANISM_COLORS[_mech], alpha=0.8)
                _saxi.set(ylim=targety_bounds, xlim=targetx_bounds, ylabel=targety, xlabel=targetx_label)
                _saxi.label_outer()
            plt.show()

        sizes = scale_markers(df_prop['iSFE/Gb'], (10, 100), between_q1q3=False)

        # x-axis: sqrt grain size
        df_prop['sqrtGS'] = np.sqrt(df_prop['grain_size'])
        df_prop['sqrtGS'] = 1 / df_prop['sqrtGS']

        subplots_target1_vs2_by_cluster_mech(
            df_prop, 'sqrtGS', target, sizes, (0, 1), target_bounds, '1 / GS^0.5 [1 / um^0.5]', no_sy)

        # x-axis: yield strength
        subplots_target1_vs2_by_cluster_mech(
            df_prop, 'yield_strength', target, sizes, (0, 800), target_bounds, 'YS [MPa]', no_sy)

    if plot_sfe_tem_vs_tc_vs_xrd:
        plot_sfe_tem_vs_tc_and_xrd()


if __name__ == "__main__":
    main()
