import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from copy import deepcopy
import matplotlib.patches as patches
from matplotlib.lines import Line2D


color_schemes = {
    'two_color_blue_green':    ["#38bae2", "#4eb156"],
    'two_color_blue_red':      ["#7aadd1", "#df5e5f"],
    'two_color_blue_red_light':["#7aadd130", "#df5e5f30"],
    'three_color_america':     ["#f7f7f7", "#6daedb", "#ffb6c2"],
    'three_color_primary':     ["#ff7f0f", "#2ba02b", "#9467bd"],
    'six_color': [
        (0.216, 0.494, 0.722, 0.7),
        (1.0,   0.498, 0.0,   0.7),
        (0.302, 0.686, 0.29,  0.7),
        (0.969, 0.506, 0.749, 0.7),
        (0.596, 0.306, 0.639, 0.7),
        (0.894, 0.102, 0.11,  0.7),
    ],
    'twelve_color': [
        (0.216, 0.494, 0.722, 0.7),
        (1.0,   0.498, 0.0,   0.7),
        (0.302, 0.686, 0.29,  0.7),
        (0.969, 0.506, 0.749, 0.7),
        (0.596, 0.306, 0.639, 0.7),
        (0.894, 0.102, 0.11,  0.7),
        (0.216, 0.494, 0.722, 0.7),
        (1.0,   0.498, 0.0,   0.7),
        (0.302, 0.686, 0.29,  0.7),
        (0.969, 0.506, 0.749, 0.7),
        (0.596, 0.306, 0.639, 0.7),
        (0.894, 0.102, 0.11,  0.7),
    ],
    'eight_color': [
        (0.216, 0.494, 0.722, 0.7),  # blue
        (1.0,   0.498, 0.0,   0.7),  # orange
        (0.302, 0.686, 0.29,  0.7),  # green
        (0.969, 0.506, 0.749, 0.7),  # pink
        (0.596, 0.306, 0.639, 0.7),  # purple
        (0.894, 0.102, 0.11,  0.7),  # red
        (0.651, 0.337, 0.157, 0.7),  # brown
        (0.576, 0.471, 0.376, 0.7),  # tan/earth
    ],
}

markers = [".", "v", "^", "s", "*", "x"]


def get_or_none(d, key):
    """Return d[key] if key is present, else None."""
    return d.get(key)


def plot_bar(ax, x_groups, y_values, y_errors, labels, formatting):
    """Create a grouped bar plot.

    Args:
        ax: Matplotlib axes
        x_groups: List of group indices (e.g. [1,2,3,1,2,3,...])
        y_values: Corresponding bar heights
        y_errors: Corresponding error bar values
        labels: Dict mapping group index → display label
        formatting: Dict with keys:
            style_size: 'paper' or 'presentation'
            color_palette: palette name from color_schemes, or '#rrggbb'
            bar_width: float (default 0.25)
            horizontal: bool
            edgecolor: color or None
            hatch: hatch pattern string
            color_shift: int, skip first N colors
            extra_labels: Dict {group_idx: [label_per_bar]}
            extra_y_shift / extra_x_shift: float offsets for extra labels
            label_rotation: degrees (default 20)
            format_string: callable mapping label value → display string
            per_group_labels: list of tick labels
    """
    label_size = 14 if formatting['style_size'] == 'presentation' else 10

    num_groups        = len(set(x_groups))
    max_bars_per_group = max(Counter(x_groups).values())
    ordered_groups    = sorted(set(x_groups))

    values_by_group = {}
    errors_by_group = {}
    for i in range(len(x_groups)):
        g = x_groups[i]
        values_by_group.setdefault(g, []).append(y_values[i])
        errors_by_group.setdefault(g, []).append(y_errors[i])

    if 'color_palette' not in formatting:
        formatting['color_palette'] = 'six_color'

    if formatting['color_palette'].startswith('#'):
        colors = [formatting['color_palette']]
        assert num_groups == 1
    else:
        assert formatting['color_palette'] in color_schemes
        colors = color_schemes[formatting['color_palette']]
        if 'color_shift' in formatting:
            colors = colors[formatting['color_shift']:]
        colors = colors[:num_groups]

    bar_width = formatting.get('bar_width', 0.25)
    hatch     = formatting.get('hatch', '')
    x         = np.arange(max_bars_per_group)

    for i, group_num in enumerate(ordered_groups):
        if formatting.get('horizontal'):
            bars = ax.barh(
                x + group_num * bar_width,
                values_by_group[group_num],
                height=bar_width,
                label=labels[group_num],
                edgecolor=get_or_none(formatting, 'edgecolor'),
                color=colors[i],
                xerr=errors_by_group[group_num],
                hatch=hatch,
            )
        else:
            bars = ax.bar(
                x + group_num * bar_width,
                values_by_group[group_num],
                width=bar_width,
                label=labels[group_num],
                edgecolor=get_or_none(formatting, 'edgecolor'),
                color=colors[i],
                yerr=errors_by_group[group_num],
                hatch=hatch,
            )

        extra = formatting.get('extra_labels', {})
        if group_num in extra:
            x_shift = formatting.get('extra_x_shift', 0)
            y_shift = formatting.get('extra_y_shift', 0)
            fmt_fn  = formatting.get('format_string', str)
            for j, bar in enumerate(bars):
                lbl_text = fmt_fn(extra[group_num][j])
                if formatting.get('horizontal'):
                    ax.text(
                        x_shift,
                        bar.get_y() + bar.get_height() / 2 + y_shift,
                        lbl_text,
                        ha='center', va='bottom', fontsize=label_size, color='black',
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2 + x_shift,
                        bar.get_height() + y_shift,
                        lbl_text,
                        ha='center', va='bottom', fontsize=label_size, color='black',
                    )

    label_rotation = formatting.get('label_rotation', 20)
    if 'per_group_labels' in formatting:
        if formatting.get('horizontal'):
            ax.set_yticks(x + bar_width * (num_groups - 1) / 2)
            ax.set_yticklabels(formatting['per_group_labels'], fontsize=14, rotation=label_rotation)
        else:
            ax.set_xticks(x + bar_width * (num_groups - 1) / 2)
            ax.set_xticklabels(formatting['per_group_labels'], fontsize=14, rotation=label_rotation)


def plot_zero_one_matrix(ax, matrix, row_labels, formatting):
    """Plot a binary matrix as filled/empty ellipses.

    Args:
        ax: Matplotlib axes
        matrix: 2D array of 0s and 1s
        row_labels: Label strings for each row
        formatting: Dict with keys:
            style_size: 'paper' or 'presentation'
            label_x / label_y: label anchor position
            x_start / y_start: first circle position
            x_width / y_width: spacing between circles
            circle_width / circle_height: ellipse dimensions
    """
    font_size = 18 if formatting['style_size'] == 'presentation' else 14

    for i, lbl in enumerate(row_labels):
        ax.text(
            formatting['label_x'],
            formatting['y_start'] + i * formatting['y_width'],
            lbl, color='black', ha='right', va='center', fontsize=font_size,
        )

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            facecolor = '#222222' if matrix[i][j] == 0 else '#EEEEEE'
            rect = patches.Ellipse(
                (formatting['x_start'] + j * formatting['x_width'],
                 formatting['y_start'] + i * formatting['y_width']),
                formatting['circle_width'], formatting['circle_height'],
                linewidth=2, facecolor=facecolor, clip_on=False,
            )
            ax.add_patch(rect)


def plot_line(ax, x_values, y_values, y_confidence, labels, formatting):
    """Create a line plot with shaded confidence intervals.

    Args:
        ax: Matplotlib axes
        x_values: List of x arrays (one per line)
        y_values: List of y arrays (one per line)
        y_confidence: List of half-width arrays for shaded bands
        labels: List of label strings (one per line)
        formatting: Dict with keys:
            color_palette: palette name or '#rrggbb'
            color_shift: int, skip first N colors
            linewidth: float (default 0.6)
            linestyle: str or list of str (default '-')
    """
    if formatting['color_palette'].startswith('#'):
        colors = [formatting['color_palette']] * len(x_values)
    else:
        assert formatting['color_palette'] in color_schemes
        colors = color_schemes[formatting['color_palette']]
        if 'color_shift' in formatting:
            colors = colors[formatting['color_shift']:]

    linewidth  = formatting.get('linewidth', 0.6)
    ls_setting = formatting.get('linestyle', '-')
    linestyles = ls_setting if isinstance(ls_setting, list) else [ls_setting] * len(x_values)

    for i in range(len(x_values)):
        ax.plot(x_values[i], y_values[i], label=labels[i],
                linewidth=linewidth, color=colors[i], linestyle=linestyles[i])
        ax.fill_between(
            x_values[i],
            np.array(y_values[i]) - np.array(y_confidence[i]),
            np.array(y_values[i]) + np.array(y_confidence[i]),
            alpha=0.2, color=colors[i],
        )


def plot_scatter(ax, x_values, y_values, formatting):
    """Create a scatter plot.

    Args:
        ax: Matplotlib axes
        x_values: List of x arrays (one per series)
        y_values: List of y arrays (one per series)
        formatting: Dict with keys:
            color_palette: palette name or '#rrggbb'
            size: marker size (default 5)
            marker: list of marker strings (default None per series)
    """
    if formatting['color_palette'].startswith('#'):
        colors = [formatting['color_palette']]
        assert len(x_values) == 1
    else:
        assert formatting['color_palette'] in color_schemes
        colors = color_schemes[formatting['color_palette']]

    size   = formatting.get('size', 5)
    marker = formatting.get('marker', [None] * len(x_values))

    for i in range(len(x_values)):
        ax.scatter(x_values[i], y_values[i], color=colors[i], s=size, marker=marker[i])


def plot_box_whisker(ax, data, labels, formatting):
    """Create a horizontal notched box-and-whisker plot.

    Args:
        ax: Matplotlib axes
        data: List of value arrays (one per box)
        labels: List of label strings
        formatting: Dict with keys:
            color_palette: palette name or '#rrggbb'
    """
    if formatting['color_palette'].startswith('#'):
        colors = [formatting['color_palette']]
    else:
        assert formatting['color_palette'] in color_schemes
        colors = color_schemes[formatting['color_palette']]

    ax.boxplot(data, labels=labels, patch_artist=True, notch=True, vert=False,
               boxprops=dict(facecolor=colors[0]), showfliers=False)


def plot_kde(ax, data, labels, formatting):
    """Create overlapping KDE density plots.

    Args:
        ax: Matplotlib axes
        data: List of value arrays (one per distribution)
        labels: List of label strings
        formatting: Dict with keys:
            color_palette: palette name or '#rrggbb'
    """
    if formatting['color_palette'].startswith('#'):
        colors = [formatting['color_palette']]
    else:
        assert formatting['color_palette'] in color_schemes
        colors = color_schemes[formatting['color_palette']]

    for i in range(len(data)):
        sns.kdeplot(data[i], label=labels[i], fill=True, color=colors[i], ax=ax)


def plot_text(ax, text, x, y, formatting):
    """Render text at a given axes position.

    Args:
        ax: Matplotlib axes
        text: String to display
        x / y: Position
        formatting: Dict with keys:
            color_palette: text color
            fontsize: int
    """
    ax.text(x, y, text, color=formatting['color_palette'],
            fontsize=formatting['fontsize'], ha='center')


def create_axes(
    plot_dimensions,
    formatting,
    x_labels=None,
    y_labels=None,
    titles=None,
    sup_x_label="",
    sup_x_label_shift=0.01,
    sup_y_label="",
    sup_title="",
):
    """Create a figure and 2D axes grid with labels and styling.

    Args:
        plot_dimensions: (rows, cols) tuple
        formatting: Dict with keys:
            style_size: 'paper' or 'presentation'
            figsize: (width, height) in inches
            label_size / title_size / tick_size: int overrides
            x_lim / y_lim: [rows][cols] limits
            x_ticks / y_ticks: [rows][cols] of ([positions], [labels])
            hide_spines: bool, hide top/right spines
            separate_spines: bool, offset bottom/left spines
            has_grid / has_x_grid / has_y_grid: bool
        x_labels / y_labels / titles: [rows][cols] label strings
        sup_x_label / sup_y_label / sup_title: figure-level text
        sup_x_label_shift: vertical nudge for sup_x_label

    Returns:
        fig, ax (2D list of axes)
    """
    fig, ax = plt.subplots(plot_dimensions[0], plot_dimensions[1], figsize=formatting['figsize'])

    default = [["" for _ in range(plot_dimensions[1])] for _ in range(plot_dimensions[0])]
    if x_labels is None: x_labels = deepcopy(default)
    if y_labels is None: y_labels = deepcopy(default)
    if titles   is None: titles   = deepcopy(default)

    # Normalise axes to always be [[ax]] shape
    if plot_dimensions[0] == plot_dimensions[1] == 1:
        ax = [[ax]]
    elif plot_dimensions[0] == 1:
        ax = [ax]
    elif plot_dimensions[1] == 1:
        ax = [[a] for a in ax]

    label_size = 14 if formatting['style_size'] == 'presentation' else 10
    title_size = 14 if formatting['style_size'] == 'presentation' else 14
    tick_size  = 14 if formatting['style_size'] == 'presentation' else 10

    label_size = formatting.get('label_size', label_size)
    title_size = formatting.get('title_size', title_size)
    tick_size  = formatting.get('tick_size',  tick_size)

    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            ax[i][j].set_xlabel(x_labels[i][j], fontsize=label_size)
            ax[i][j].set_ylabel(y_labels[i][j], fontsize=label_size)
            ax[i][j].set_title(titles[i][j],    fontsize=title_size)
            ax[i][j].tick_params(axis='x', labelsize=tick_size)
            ax[i][j].tick_params(axis='y', labelsize=tick_size)

            if 'x_lim'   in formatting: ax[i][j].set_xlim(formatting['x_lim'][i][j])
            if 'y_lim'   in formatting: ax[i][j].set_ylim(formatting['y_lim'][i][j])
            if 'x_ticks' in formatting:
                ax[i][j].set_xticks(formatting['x_ticks'][i][j][0])
                ax[i][j].set_xticklabels(formatting['x_ticks'][i][j][1], fontsize=tick_size)
            if 'y_ticks' in formatting:
                ax[i][j].set_yticks(formatting['y_ticks'][i][j][0])
                ax[i][j].set_yticklabels(formatting['y_ticks'][i][j][1], fontsize=tick_size)

            if formatting.get('hide_spines'):
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].spines['right'].set_visible(False)

            if formatting.get('separate_spines'):
                ax[i][j].spines['left'].set_position(('outward', 5))
                ax[i][j].spines['bottom'].set_position(('outward', 5))

            if get_or_none(formatting, 'has_grid'):
                ax[i][j].grid()
            if get_or_none(formatting, 'has_x_grid'):
                ax[i][j].grid(axis='x', linestyle='--', alpha=0.7)
            if get_or_none(formatting, 'has_y_grid'):
                ax[i][j].grid(axis='y', linestyle='--', alpha=0.7)

    fig.supxlabel(sup_x_label, y=sup_x_label_shift, fontsize=label_size)
    fig.supylabel(sup_y_label, fontsize=label_size)
    fig.suptitle(sup_title)

    return fig, ax


def create_legend(fig, ax, plot_dimensions, formatting):
    """Add a legend to a figure or individual axes.

    Args:
        fig: Matplotlib figure
        ax: 2D list of axes (from create_axes)
        plot_dimensions: (rows, cols) tuple
        formatting: Dict with keys:
            style_size: 'paper' or 'presentation'
            fontsize: int override
            type: 'is_global' (figure-level) or 'is_local' (per-axes)
            show_point: bool, add dot markers to legend lines
            loc / ncol / bbox_to_anchor: standard legend kwargs
            extra_handles: list of (handle, label) pairs to append

    Returns:
        fig, ax
    """
    legend_size = 14 if formatting['style_size'] == 'presentation' else 10
    legend_size = formatting.get('fontsize', legend_size)

    if formatting['type'] == 'is_global':
        handles, labels = ax[0][0].get_legend_handles_labels()
        if 'extra_handles' in formatting:
            for h, lbl in formatting['extra_handles']:
                handles.append(h)
                labels.append(lbl)

        if get_or_none(formatting, 'show_point'):
            custom_lines = [
                Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(), marker=markers[i])
                for i, h in enumerate(handles)
            ]
            fig.legend(custom_lines, labels,
                       loc=formatting['loc'], ncol=formatting['ncol'],
                       bbox_to_anchor=formatting['bbox_to_anchor'], fontsize=legend_size)
        else:
            fig.legend(handles, labels,
                       loc=formatting['loc'], ncol=formatting['ncol'],
                       bbox_to_anchor=formatting['bbox_to_anchor'], fontsize=legend_size)

    elif formatting['type'] == 'is_local':
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                if get_or_none(formatting, 'show_point'):
                    handles, labels = ax[i][j].get_legend_handles_labels()
                    custom_lines = [
                        Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(), marker='o')
                        for h in handles
                    ]
                    ax[i][j].legend(custom_lines, labels,
                                    loc=formatting['loc'], ncol=formatting['ncol'],
                                    bbox_to_anchor=formatting['bbox_to_anchor'], fontsize=legend_size)
                else:
                    ax[i][j].legend(loc=formatting['loc'], ncol=formatting['ncol'],
                                    bbox_to_anchor=formatting['bbox_to_anchor'], fontsize=legend_size)

    return fig, ax