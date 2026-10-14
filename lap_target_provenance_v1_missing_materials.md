# Draft request — автору исходного mRKS-экспорта (НЕ ОТПРАВЛЕН)

Нужно аутентифицировать 90 записей, прошедших в data_vxc_train.pickle
(SHA256 0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49).
Для минимального подтверждения достаточно исходника/config и связанных логов
H2, BeH2, CO плюс подтверждения общей процедуры и исключений для остальных 87.

1. Пришлите script/notebook и run config, создававшие inp_mrks.npz, особенно assignments для vxc_grid, exc_wf, t_wf и meta_keys. В NPZ input_file указан /home/mmedvedev/lsvvt/chep/ready/{system}/inp.orca.zst. Нужны версия реализации inversion/mRKS и связь outputs с нашими files/hashes. Это закроет producer→NPZ и физический смысл potential/energy fields; публикация об общем алгоритме не заменяет данный exporter.

2. Пришлите script/формулу, создававшие H5 E_xc, E_cbs_1 и E_no_xc до prepare_vxc.py. Сохранился h5_vrho_from_mrks_sanity.csv. На пилоте E_xc=E_cbs_1−E_no_xc и F32 импорт дают наши legacy targets. Подтвердите, содержит ли E_no_xc E_nuc+Tr(P_KS T)+Tr(P_KS V_ext)+0.5Tr(P_KS J_KS), какая именно плотность/геометрия/заряд использованы и когда округлялись значения. Это закроет рецепт, не подменяя его точным E_xc[n_KS].

3. Укажите method/basis/extrapolation для E_CBS_1 и что означают E_scf_5Z, E_scf_QZ, E_corr_5Z, E_corr_QZ в extrapolation_e.csv; включены ли ядра/эмпирические поправки. Нужны связанный расчетный input/log или краткая первичная формула автора. Это закроет происхождение reference total/electronic energy и CBS приближения.

4. Укажите правило gauge и точный момент его применения к сохранённому vxc_grid: potential shift, orbital-energy shift, состояние итерации, units/spin, masking/filtering/tail clipping/interpolation. HOMO+i_ekt (Ha) равен +5.8980e-5 для H2, −3.08894e-6 для BeH2, −1.08943e-4 для CO; почему это не точный ноль и согласованы ли орбитали с экспортированным потенциалом? Assignment плюс соответствующий export log снимет GAUGE_UNRESOLVED.

5. Для dispersions_mrks.pickle, добавленного в Git 4ed868e (SHA256 a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440), пришлите исходные H5 atom/charge/spin или geometry hashes, команду --h5-dir, backend/library version, фактические PBE0/D3(BJ) numerical parameters и настройки дополнительных членов. Это закроет геометрию и историческую конфигурацию, которые не устанавливаются современным D3 replay.

6. Подтвердите intended energy convention: NN integral представляет полный XC либо остаток E_target−D? Был ли исходный E_target чистым ab initio XC? Сейчас код добавляет D к prediction один раз, сохраняя E_target. Краткое первичное описание этой договорённости закроет semantic blocker; мы не хотим оправдывать ее задним числом.

Личность автора не установлена; имя и адрес получателя не предполагаются.
Никакие targets, potentials, loss или архитектура не изменены.
