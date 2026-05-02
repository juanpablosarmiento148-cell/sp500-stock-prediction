import json

# Read the file
with open('sp500_stock_prediction.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()
nb = json.loads(content)

# Find and modify the cell after Plot 4
cells = nb['cells']
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'plt.show()' in src and i > 0:
            # Insert new Matrix cell after this one
            matrix_cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    "# Plot 5: Matrix-Themed Correlation Heatmap",
                    "plt.style.use('dark_background')",
                    "",
                    "fig, ax = plt.subplots(figsize=(14, 10))",
                    "corr_matrix = df_clean[feature_columns + ['Target']].corr()",
                    "",
                    "im = ax.imshow(corr_matrix.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)",
                    "",
                    "ax.set_xticks(range(len(corr_matrix.columns)))",
                    "ax.set_yticks(range(len(corr_matrix.columns)))",
                    "ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9, color='#00ff00')",
                    "ax.set_yticklabels(corr_matrix.columns, fontsize=9, color='#00ff00')",
                    "",
                    "for spine in ax.spines.values():",
                    "    spine.set_edgecolor('#00ff00')",
                    "    spine.set_lineWidth(1.5)",
                    "",
                    "for i in range(len(corr_matrix)):",
                    "    for j in range(len(corr_matrix)):",
                    "        val = corr_matrix.iloc[i, j]",
                    "        color = '#00ff00'",
                    "        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')",
                    "",
                    "cbar = plt.colorbar(im, ax=ax, shrink=0.8)",
                    "cbar.ax.tick_params(colors='#00ff00', labelsize=9)",
                    "cbar.set_label('Correlation', color='#00ff00', fontsize=12)",
                    "",
                    "ax.set_title('FEATURE CORRELATION MATRIX\\nThe Matrix of Market Relationships', ",
                    "             fontsize=16, fontweight='bold', color='#00ff00', pad=20)",
                    "",
                    "ax.tick_params(axis='x', colors='#00ff00')",
                    "ax.tick_params(axis='y', colors='#00ff00')",
                    "",
                    "fig.patch.set_facecolor('#0a0a0a')",
                    "plt.tight_layout()",
                    "plt.show()",
                    "",
                    "plt.style.use('seaborn-v0_8-whitegrid')"
                ]
            }
            cells.insert(i + 1, matrix_cell)
            break

# Write back
with open('sp500_stock_prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print('Done')