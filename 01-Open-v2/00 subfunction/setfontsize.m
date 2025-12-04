function setfontsize(ftsz, h)

if nargin == 1

    set(get(gca, 'XLabel'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(get(gca, 'YLabel'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(get(gca, 'Title'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(gca, 'FontSize', ftsz);
    set(gca, 'Fontname', 'Times New Roman');
else
    set(get(h, 'XLabel'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(get(h, 'YLabel'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(get(h, 'Title'), 'FontSize', ftsz, 'Fontname', 'Times New Roman');
    set(h, 'FontSize', ftsz);
    set(h, 'Fontname', 'Times New Roman');
end
