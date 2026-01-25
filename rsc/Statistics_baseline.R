#### 统计聚类特征
rm(list = ls())
# 加载必要的包
library(tableone)
library(officer)
library(flextable)
library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)
library(ggpubr)
library(patchwork)
library(RColorBrewer)
library(cowplot)

### 数据准备 2025-12-22, 改deepseek
if (TRUE) {
  load("data/statistics_data_plus_2025-12-14.RData")

  par(family = "Arial")

  df <- df_onset %>% #  2920中重复肾活检6例，2913, 数缺失 1，
    # filter(eGFR_category_onset %in% c("G1", "G2", "G3a", "G3b")) %>% # 239
    filter(rowSums(select(., starts_with("malignancy"))) == 0) %>%    # 124
    mutate(
      up24 = ifelse(up24 > 100, up24 / 1000, up24)
    ) %>%
    rename(
      qwen_pca = kmeans_pca_qwen,
      deepseek_pca = kmeans_pca_deepseek,
    ) %>%
    mutate(
      qwen_pca = factor(qwen_pca, levels = c(1, 0),
                        labels = c("low-activity", "high-activity")),
      # qwen_tsne = factor(qwen_tsne),
      deepseek_pca = factor(deepseek_pca, levels = c(0, 1),
                            labels = c("low-activity", "high-activity"))
    ) %>%
    mutate(
      immuno = rowSums(across(c(fk506, mmf, ctx, aza, leflunomide))) > 0
    ) %>%
    mutate(
      rapid_up24 = case_when(
        up24 >= 0.5 ~ "Yes",
        up24 < 0.5 ~ "No"
      ),
      rapid_up24 = factor(rapid_up24, levels = c("No", "Yes"))
    ) %>%
    mutate(
      steroid = factor(steroid, levels = c(0, 1),
                       labels = c("No", "Yes")),
      rasb = factor(rasb, levels = c(0, 1),
                    labels = c("No", "Yes")),
      mmf = factor(mmf, levels = c(0, 1),
                   labels = c("No", "Yes")),
      immuno = factor(immuno, levels = c(FALSE, TRUE),
                      labels = c("No", "Yes")),
      leflunomide = factor(leflunomide, levels = c(0, 1),
                           labels = c("No", "Yes")),
      fk506 = factor(fk506, levels = c(0, 1),
                     labels = c("No", "Yes")),
      ctx = factor(ctx, levels = c(0, 1),
                   labels = c("No", "Yes")),
      aza = factor(aza, levels = c(0, 1),
                   labels = c("No", "Yes"))
    )

  df$qwen_pca <- df$deepseek_pca
  if (!dir.exists("results")) {
    dir.create("results")
  }

}

#### baseline Table FOR CLUSTERS
if (TRUE) {
  demographic_vars <- c("age", "Gender", "systolic_bp", "diastolic_bp")


  diagnosis_vars <- c(
    "hypertentsion", "diabetic", "rapid_up24"
  )

  # 实验室指标（选择关键指标）
  lab_vars <- c(
    "s_alb", "s_cr", "s_ua", "s_hb", "up24", "eGFR_onset", "eGFR_category_onset"
  )

  # 病理指标
  pathology_vars <- c(
    "MEST_C_score_M", "MEST_C_score_E", "MEST_C_score_S",
    "MEST_C_score_T", "MEST_C_score_C"
  )

  # 治疗药物（选择主要治疗药物）
  medication_vars <- c(
    "steroid", "rasb", "mmf", "ctx", "fk506", "aza", 'leflunomide', 'immuno'
  )

  # 合并所有变量
  all_vars <- c(demographic_vars, diagnosis_vars, lab_vars, pathology_vars, medication_vars)

  # 定义分类变量
  categorical_vars <- c(
    "Gender", "eGFR_category_onset", "MEST_C_score_M", "MEST_C_score_E", "MEST_C_score_S",
    "MEST_C_score_T", "MEST_C_score_C", "steroid", "rasb", "mmf", "ctx", "fk506", "aza",
    "hypertentsion", "diabetic", "rapid_up24", 'leflunomide', 'immuno'
  )


  # 创建TableOne对象
  table_one <- CreateTableOne(
    vars = all_vars,
    data = df,
    factorVars = categorical_vars
  )

  # 打印表格
  table_one_print <- print(
    table_one,
    nonnormal = c("s_cr", "eGFR_onset", "up24"),
    exact = categorical_vars,
    quote = FALSE,
    noSpaces = TRUE,
    printToggle = FALSE,
    showAllLevels = FALSE,
    contDigits = 1,  # 连续变量小数位数
    catDigits = 1,   # 分类变量小数位数
    pDigits = 3      # p值小数位数
  )

  convert_row_names <- function(table_df) {
    rownames_vec <- rownames(table_df)
    new_rownames <- character(length(rownames_vec))

    for (i in seq_along(rownames_vec)) {
      rname <- rownames_vec[i]

      if (rname == "n") {
        new_rownames[i] <- "n"
      } else if (rname == "age (mean (SD))") {
        new_rownames[i] <- "Age (years)"
      } else if (rname == "Gender = 女 (%)") {
        new_rownames[i] <- "Female"
      } else if (rname == "systolic_bp (mean (SD))") {
        new_rownames[i] <- "Systolic blood pressure (mmHg)"
      } else if (rname == "diastolic_bp (mean (SD))") {
        new_rownames[i] <- "Diastolic blood pressure (mmHg)"
      } else if (rname == "hypertentsion = 1 (%)") {
        new_rownames[i] <- "Hypertension"
      } else if (rname == "diabetic = 1 (%)") {
        new_rownames[i] <- "Diabetes"
      } else if (rname == "rapid_up24 = Yes (%)") {
        new_rownames[i] <- "high-risk IgAN"
      } else if (rname == "s_alb (mean (SD))") {
        new_rownames[i] <- "Serum albumin (g/L)"
      } else if (rname == "s_cr (mean (SD))" | rname == "s_cr (median [IQR])") {
        new_rownames[i] <- "Serum creatinine (μmol/L)"
      } else if (rname == "s_ua (mean (SD))") {
        new_rownames[i] <- "Serum uric acid (μmol/L)"
      } else if (rname == "s_hb (mean (SD))") {
        new_rownames[i] <- "Hemoglobin (g/L)"
      } else if (rname == "up24 (mean (SD))" | rname == "up24 (median [IQR])") {
        new_rownames[i] <- "24-hour urine protein (g/24h)"
      } else if (rname == "eGFR_onset (mean (SD))" | rname == "eGFR_onset (median [IQR])") {
        new_rownames[i] <- "eGFR (ml/min/1.73m²)"
      } else if (rname == "eGFR_category_onset (%)") {
        new_rownames[i] <- "eGFR category"
      } else if (grepl("^   G", rname)) {
        # 这些是eGFR分级的子项，保持原样在后面处理
        new_rownames[i] <- rname
      } else if (rname == "s_iga (mean (SD))") {
        new_rownames[i] <- "Serum IgA (g/L)"
      } else if (rname == "s_igg (mean (SD))") {
        new_rownames[i] <- "Serum IgG (g/L)"
      } else if (rname == "s_igm (mean (SD))") {
        new_rownames[i] <- "Serum IgM (g/L)"
      } else if (rname == "MEST_C_score_M = 1 (%)") {
        new_rownames[i] <- "MEST-C M1"
      } else if (rname == "MEST_C_score_E = 1 (%)") {
        new_rownames[i] <- "MEST-C E1"
      } else if (rname == "MEST_C_score_S = 1 (%)") {
        new_rownames[i] <- "MEST-C S1"
      } else if (rname == "MEST_C_score_T (%)") {
        new_rownames[i] <- "MEST-C T score"
      } else if (grepl("^   [0-2]", rname)) {
        # 这些是T分期的子项
        new_rownames[i] <- rname
      } else if (rname == "MEST_C_score_C = 1 (%)") {
        new_rownames[i] <- "MEST-C C1"
      } else if (rname == "steroid = Yes (%)") {
        new_rownames[i] <- "Corticosteroid therapy"
      } else if (rname == "rasb = Yes (%)") {
        new_rownames[i] <- "RAS blockade therapy"
      } else if (rname == "mmf = Yes (%)") {
        new_rownames[i] <- "Mycophenolate mofetil"
      } else if (rname == "ctx = Yes (%)") {
        new_rownames[i] <- "Cyclophosphamide"
      } else if (rname == "fk506 = Yes (%)") {
        new_rownames[i] <- "Tacrolimus"
      } else if (rname == "aza = Yes (%)") {
        new_rownames[i] <- "Azathioprine"
      } else if (rname == "leflunomide = Yes (%)") {
        new_rownames[i] <- "Leflunomide"
      } else if (rname == "immuno = Yes (%)") {
        new_rownames[i] <- "Immunosuppressive therapy"
      } else {
        new_rownames[i] <- rname
      }
    }

    return(new_rownames)
  }

  # 应用行名转换
  new_rownames <- convert_row_names(table_one_print)
  table_df <- as.data.frame(table_one_print)
  rownames(table_df) <- new_rownames

  # 创建TableOne对象
  table_one_qwen <- CreateTableOne(
    vars = all_vars,
    data = df,
    factorVars = categorical_vars,
    strata = 'qwen_pca'
  )

  # 打印表格
  table_qwen_print <- print(
    table_one_qwen,
    nonnormal = c("s_cr", "eGFR_onset", "up24"),
    quote = FALSE,
    noSpaces = TRUE,
    printToggle = FALSE,
    showAllLevels = FALSE,
    contDigits = 1,  # 连续变量小数位数
    catDigits = 1,   # 分类变量小数位数
    pDigits = 3      # p值小数位数
  )

  # 应用行名转换
  new_rownames <- convert_row_names(table_qwen_print)
  table_df_qwen <- as.data.frame(table_qwen_print)
  rownames(table_df_qwen) <- new_rownames
  table_full_qwen <- cbind(table_df, table_df_qwen) %>%
    select(-'test') %>%
    rename("p value" = "p")
  # 创建flextable对象 - 三线表格式
  # 计算列数（如果尚未计算）
  ncols <- ncol(table_full_qwen)

  flex_table <- table_full_qwen %>%
    tibble::rownames_to_column(var = "Characteristics") %>%
    flextable() %>%
    fontsize(size = 8, part = "all") %>%
    flextable::font(fontname = "Times New Roman", part = "all") %>%
    border_remove() %>%
    hline_top(border = fp_border(width = 1.5), part = "header") %>%
    hline(i = 1, border = fp_border(width = 1.5), part = "header") %>%
    hline_bottom(border = fp_border(width = 1.5), part = "body") %>%
    align(align = "center", part = "header") %>%
    align(align = "center", part = "body") %>%
    align(j = 1, align = "left", part = "body") %>%
    bold(i = 1, bold = TRUE, part = "header") %>%
    line_spacing(space = 0.7, part = "body") %>%
    line_spacing(space = 0.7, part = "header") %>%
    autofit() %>%                              # 先自适应，再固定宽度
    set_table_properties(layout = "fixed") %>% # 固定布局，便于精确控制宽度
    width(j = 1, width = 1.5) %>%              # 第一列较宽（特征名）
    width(j = 2:ncols, width = 1.0)            # 其它列宽减小以缩小间距


  # 创建word文档
  doc <- read_docx() %>%
    # 添加表格标题
    body_add_par("Table 1: Baseline Characteristics of IgA Nephropathy Patients by Clusters",
                 style = "table title") %>%
    # 添加表格
    body_add_flextable(flex_table) %>%
    # 添加注释
    body_add_par("Note: Continuous variables are presented as mean (SD) or median [IQR]; categorical variables are presented as frequency (%). eGFR, estimated glomerular filtration rate; MEST-C, Oxford classification of IgA nephropathy; RAS, renin-angiotensin system; Immunosuppressive therapy was defined as initial administration of any immunosuppressive agent - including mycophenolate mofetil, cyclophosphamide, tacrolimus, azathioprine, or leflunomide - within one year following renal biopsy.",
                 style = "Normal") %>%
    # 设置页面布局
    body_end_section_continuous()  # 连续分节，确保表格不跨页

  # 保存文档
  print(doc, target = "results/IgAN_Baseline_Table_clusters.docx")
}
#### 作图
if (TRUE) {
  # 设置Arial字体（如果系统中有）
  windowsFonts(Arial = windowsFont("Arial"))
  theme_set(theme_minimal(base_family = "Arial"))

  # 自定义主题函数
  theme_arial_publication <- function(base_size = 10) {
    theme_minimal(base_family = "Arial", base_size = base_size) %+replace%
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5,
                                  margin = margin(b = 10)),
        axis.title = element_text(face = "bold"),
        axis.text = element_text(color = "black"),
        legend.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"),
        panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.5),
        strip.background = element_rect(fill = "grey95", color = "grey70"),
        plot.margin = margin(10, 10, 10, 10)
      )
  }

  # 1. 准备数据 - 将两个分类指标合并
  df <- df %>%
    mutate(
      rapid_up24 = ifelse(rapid_up24 == "Yes", "high-risk", "low-risk"),
      rapid_up24 = factor(rapid_up24, levels = c("low-risk", "high-risk")),
      qwen_pca = factor(qwen_pca),
      combined_group = interaction(qwen_pca, rapid_up24, sep = " | "),
      # 创建用于分面的组别
      cluster_label = paste("Subtype:", qwen_pca),
      risk_label = paste("Risk:", rapid_up24)
    )

  # 2. 关键临床指标的交互可视化
  # 定义要展示的关键临床指标
  key_clinical_vars <- c(
    "age", "systolic_bp", "diastolic_bp",
    "s_alb", "s_cr", "eGFR_onset", "up24"
  )

  key_clinical_labels <- c(
    "Age (years)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)",
    "Serum albumin (g/L)", "Serum creatinine (μmol/L)",
    "eGFR (ml/min/1.73m²)", "24h proteinuria (g/24h)"
  )

  # 准备数据
  clinical_interaction_data <- df %>%
    select(all_of(key_clinical_vars), qwen_pca, rapid_up24) %>%
    pivot_longer(
      cols = all_of(key_clinical_vars),
      names_to = "variable",
      values_to = "value"
    ) %>%
    filter(!is.na(rapid_up24)) %>%
    mutate(
      variable = factor(variable,
                        levels = key_clinical_vars,
                        labels = key_clinical_labels)
    )

  # 图1: 点图+箱线图展示双分类下的临床指标分布
  if (TRUE) {
    # 安装必要的包（如果尚未安装）
    # install.packages(c("ggplot2", "tidyverse", "gridExtra", "scales", "ggpubr", "rstatix", "plotly", "htmlwidgets"))

    library(ggplot2)
    library(tidyr)
    library(dplyr)
    library(scales)
    library(ggpubr)
    library(rstatix)
    library(patchwork)

    # 设置Arial字体
    windowsFonts(Arial = windowsFont("Arial"))
    theme_set(theme_minimal(base_family = "Arial"))

    # 自定义函数：将p值转换为星号标记
    pvalue_to_stars <- function(p) {
      if (is.na(p)) return("")
      if (p < 0.0001) return("****")
      if (p < 0.001) return("***")
      if (p < 0.01) return("**")
      if (p < 0.05) return("*")
      return("ns")
    }

    # 自定义函数：格式化p值
    format_pvalue <- function(p) {
      if (is.na(p)) return("NA")
      if (p < 0.0001) return("< 0.0001")
      if (p < 0.001) return("< 0.001")
      return(sprintf("%.3f", p))
    }

    # 1. nature标准的颜色方案
    nature_colors <- c(
      "#1A85FF", # 蓝色 - Cluster 1
      "#D41159", # 红色 - Cluster 2
      "#40B0A6", # 青色 - Cluster 3
      "#E1BE6A", # 金色 - Cluster 4
      "#0F2080"  # 深蓝 - Cluster 5（如果有）
    )[1:length(unique(df$qwen_pca))]

    # 2. 改进的主题函数
    theme_nature <- function(base_size = 10) {
      theme_minimal(base_family = "Arial", base_size = base_size) %+replace%
        theme(
          plot.title = element_text(face = "bold", hjust = 0.5,
                                    size = base_size + 2, margin = margin(b = 15)),
          plot.subtitle = element_text(hjust = 0.5, color = "gray40",
                                       size = base_size, margin = margin(b = 10)),
          axis.title = element_text(face = "bold", size = base_size + 1),
          axis.title.x = element_text(margin = margin(t = 10)),
          axis.title.y = element_text(margin = margin(r = 10)),
          axis.text = element_text(color = "black", size = base_size),
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          legend.title = element_text(face = "bold", size = base_size),
          legend.text = element_text(size = base_size - 1),
          legend.position = "bottom",
          legend.box = "horizontal",
          legend.margin = margin(t = 0, b = 5),
          strip.text = element_text(face = "bold", size = base_size,
                                    margin = margin(b = 5)),
          strip.background = element_rect(fill = "grey95", color = NA),
          panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
          panel.grid.minor = element_blank(),
          panel.border = element_rect(color = "grey80", fill = NA, linewidth = 0.5),
          panel.spacing = unit(0.8, "lines"),
          plot.margin = margin(15, 15, 15, 15),
          plot.background = element_rect(fill = "white", color = NA)
        )
    }

    # 3. 计算样本量用于标签
    sample_sizes <- clinical_interaction_data %>%
      group_by(variable, rapid_up24) %>%
      summarise(
        total_n = sum(!is.na(value)),
        .groups = 'drop'
      )

    # 创建分面标签
    facet_labels <- setNames(
      paste0(sample_sizes$rapid_up24, "\n(n=", sample_sizes$total_n, ")"),
      sample_sizes$rapid_up24
    )

    # 4. 创建改进的图1 - 基础图形
    plot1_base <- clinical_interaction_data %>%
      ggplot(aes(x = qwen_pca, y = value,
                 color = qwen_pca, fill = qwen_pca)) +
      # 数据点（带有抖动）
      geom_point(
        position = position_jitterdodge(
          jitter.width = 0.15,
          dodge.width = 0.7,
          seed = 123
        ),
        size = 0.7,
        alpha = 0.3,
        shape = 16,
        color = "gray40"
      ) +
      # 小提琴图展示分布
      geom_violin(width = 0.7, alpha = 0.2, trim = FALSE,
                  color = NA, scale = "width") +
      # 箱线图核心统计量
      geom_boxplot(width = 0.15, alpha = 0.8,
                   outlier.shape = NA,
                   color = "white",
                   position = position_dodge(0.7),
                   show.legend = FALSE) +
      # 中位数标记（菱形）
      stat_summary(
        fun = median,
        geom = "point",
        shape = 18,
        size = 1,
        color = "black",
        position = position_dodge(0.7),
        show.legend = FALSE
      ) +
      # 分面
      facet_grid(
        variable ~ rapid_up24,
        scales = "free_y",
        labeller = labeller(rapid_up24 = facet_labels),
        switch = "y"
      ) +
      scale_y_continuous(position = "right") +
      # nature颜色方案
      scale_color_manual(
        values = nature_colors,
        name = "Pathological Subtype"
      ) +
      scale_fill_manual(
        values = nature_colors,
        name = "Pathological Subtype"
      ) +
      # 坐标轴标签
      labs(
        x = NULL,
        y = NULL
      ) +
      # 应用nature主题
      theme_nature(base_size = 10) +
      # 额外的主题调整
      theme(
        legend.position = "right",
        axis.title = element_blank(),
        axis.text.x = element_text(angle = 45),
        legend.key.size = unit(0.6, "cm"),
        legend.spacing.x = unit(0.3, "cm"),
        strip.text.y = element_text(angle = 0, hjust = 0),
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 0.5, vjust = 0.5)
      )

    # 5. 改进的统计检验和p值标记 - 确保p值在同一行对齐

    # 计算每个分面的统一y轴位置（用于p值标记）
    # 方法：计算每个变量的最大y值，然后在所有风险组中使用相同的相对位置
    variable_max_values <- clinical_interaction_data %>%
      group_by(variable) %>%
      summarise(
        global_max = max(value, na.rm = TRUE) * 0.9,  # 统一增加30%的余量
        global_min = min(value, na.rm = TRUE),
        .groups = 'drop'
      )

    # 计算每个分面的具体y位置
    # 为每个变量计算统一的位置，确保在同一行
    annotation_positions <- expand.grid(
      variable = unique(clinical_interaction_data$variable),
      rapid_up24 = unique(clinical_interaction_data$rapid_up24),
      stringsAsFactors = FALSE
    ) %>%
      left_join(variable_max_values, by = "variable") %>%
      mutate(
        # p值标记的统一y位置（整体检验）
        p_value_y = global_max * 0.95,
        # 第一层两两比较的y位置
        pairwise_y1 = global_max * 1.02,
        # 第二层两两比较的y位置
        pairwise_y2 = global_max * 1.07,
        # 第三层两两比较的y位置
        pairwise_y3 = global_max * 1.12
      )

    # 初始化统计结果存储
    stat_results <- list()
    sig_annotations <- data.frame()

    # 对每个临床变量和风险组进行统计检验
    for (var in unique(clinical_interaction_data$variable)) {
      for (risk_group in unique(clinical_interaction_data$rapid_up24)) {

        # 筛选数据
        data_subset <- clinical_interaction_data %>%
          filter(variable == var, rapid_up24 == risk_group)

        # 获取聚类数量
        clusters <- unique(data_subset$qwen_pca)
        n_clusters <- length(clusters)

        if (n_clusters > 1 && nrow(data_subset) > 0) {
          # 检查是否有足够的数据
          if (all(table(data_subset$qwen_pca) >= 3)) {

            # Kruskal-Wallis检验（非参数ANOVA）
            kruskal_test <- kruskal.test(value ~ qwen_pca, data = data_subset)

            # 保存结果
            stat_results[[paste(var, risk_group, sep = "_")]] <- list(
              kruskal = kruskal_test,
              clusters = clusters,
              n_total = nrow(data_subset)
            )

            # 获取预计算的y位置
            pos <- annotation_positions %>%
              filter(variable == var, rapid_up24 == risk_group)

            # 添加整体Kruskal-Wallis检验p值（固定在统一位置）
            if (!is.na(kruskal_test$p.value)) {
              sig_annotations <- rbind(sig_annotations, data.frame(
                variable = var,
                rapid_up24 = risk_group,
                x = mean(1:n_clusters),
                y = pos$p_value_y[1],
                label = ifelse(kruskal_test$p.value < 0.001,
                               paste0("p ", format_pvalue(kruskal_test$p.value)),
                               paste0("p = ", format_pvalue(kruskal_test$p.value))),
                type = "overall",
                stringsAsFactors = FALSE
              ))
            }

            # 如果整体显著(p<0.1)，进行两两比较
            if (kruskal_test$p.value < 0.1 && n_clusters > 2) {
              # Dunn's post-hoc检验
              dunn_test <- tryCatch({
                rstatix::dunn_test(
                  value ~ qwen_pca,
                  data = data_subset,
                  p.adjust.method = "bonferroni"
                )
              }, error = function(e) NULL)

              if (!is.null(dunn_test) && any(dunn_test$p.adj < 0.05)) {
                # 保存两两比较结果
                stat_results[[paste(var, risk_group, sep = "_")]]$dunn <- dunn_test

                # 筛选显著的两两比较（按p值排序，取最显著的）
                sig_pairs <- dunn_test[dunn_test$p.adj < 0.05,] %>%
                  arrange(p.adj)

                if (nrow(sig_pairs) > 0) {
                  # 使用统一的y位置，确保在同一行
                  pairwise_y_positions <- c(pos$pairwise_y1[1],
                                            pos$pairwise_y2[1],
                                            pos$pairwise_y3[1])

                  for (i in 1:min(3, nrow(sig_pairs))) {  # 最多显示3个最显著的比较
                    group1_num <- as.numeric(gsub("Cluster ", "", sig_pairs$group1[i]))
                    group2_num <- as.numeric(gsub("Cluster ", "", sig_pairs$group2[i]))

                    sig_annotations <- rbind(sig_annotations, data.frame(
                      variable = var,
                      rapid_up24 = risk_group,
                      group1 = sig_pairs$group1[i],
                      group2 = sig_pairs$group2[i],
                      x = mean(c(group1_num, group2_num)),
                      y = pairwise_y_positions[i],
                      label = pvalue_to_stars(sig_pairs$p.adj[i]),
                      type = "pairwise",
                      stringsAsFactors = FALSE
                    ))
                  }
                }
              }
            }
          }
        }
      }
    }

    # 6. 创建图形并添加显著性标记
    plot1_final <- plot1_base

    # 检查是否有显著性标记需要添加
    if (nrow(sig_annotations) > 0) {
      # 分离整体p值和两两比较标记
      overall_ann <- sig_annotations %>% filter(type == "overall")
      pairwise_ann <- sig_annotations %>% filter(type == "pairwise")

      # 添加整体检验p值（使用geom_text确保位置一致）
      if (nrow(overall_ann) > 0) {
        plot1_final <- plot1_final +
          geom_text(
            data = overall_ann,
            aes(x = x, y = y, label = label),
            size = 3.2,
            family = "Arial",
            fontface = "italic",
            color = "gray40",
            inherit.aes = FALSE,
            hjust = 0.5,
            vjust = 0
          )
      }

      # 添加两两比较的星号和连接线
      if (nrow(pairwise_ann) > 0) {
        # 首先添加连接线
        # 按y位置分组处理连接线
        unique_y_levels <- unique(pairwise_ann$y)

        for (y_level in unique_y_levels) {
          level_data <- pairwise_ann %>% filter(y == y_level)

          for (i in 1:nrow(level_data)) {
            group1_num <- as.numeric(gsub("Cluster ", "", level_data$group1[i]))
            group2_num <- as.numeric(gsub("Cluster ", "", level_data$group2[i]))

            # 连接线在标记下方一点
            bracket_y <- y_level * 0.97

            plot1_final <- plot1_final +
              annotate(
                "segment",
                x = group1_num,
                xend = group2_num,
                y = bracket_y,
                yend = bracket_y,
                linewidth = 0.4,
                color = "gray60"
              ) +
              # 添加连接线两端的竖线
              annotate(
                "segment",
                x = group1_num,
                xend = group1_num,
                y = bracket_y * 0.99,
                yend = bracket_y,
                linewidth = 0.3,
                color = "gray60"
              ) +
              annotate(
                "segment",
                x = group2_num,
                xend = group2_num,
                y = bracket_y * 0.99,
                yend = bracket_y,
                linewidth = 0.3,
                color = "gray60"
              )
          }
        }

        # 然后添加星号标记
        plot1_final <- plot1_final +
          geom_text(
            data = pairwise_ann,
            aes(x = x, y = y, label = label),
            size = 3.5,
            family = "Arial",
            fontface = "bold",
            color = "#D41159",
            inherit.aes = FALSE,
            hjust = 0.5,
            vjust = 0.5
          )
      }
    }
    if (!dir.exists("results")) {
      dir.create("results")
    }
    library(Cairo)

    # 使用Cairo设备保存PDF
    CairoPDF("results/Figure1_Clinical_Characteristics_nature.pdf",
             width = 10,
             height = 12,
             family = "Arial")

    print(plot1_final)
    dev.off()

  }
  # 4. 病理特征的交互分析
  if (TRUE) {
    pathology_vars <- c(
      "MEST_C_score_M", "MEST_C_score_E", "MEST_C_score_S",
      "MEST_C_score_T", "MEST_C_score_C"
    )

    pathology_data <- df %>%
      select(all_of(pathology_vars), qwen_pca, rapid_up24) %>%
      filter(!is.na(rapid_up24)) %>%
      mutate(
        MEST_C_score_M = ifelse(
          MEST_C_score_M >= 1, 1, 0
        ),
        MEST_C_score_E = ifelse(
          MEST_C_score_E >= 1, 1, 0
        ),
        MEST_C_score_S = ifelse(
          MEST_C_score_S >= 1, 1, 0),
        MEST_C_score_T = ifelse(
          MEST_C_score_T >= 2, 2, MEST_C_score_T),
        MEST_C_score_C = ifelse(
          MEST_C_score_C >= 1, 1, 0)
      ) %>%
      pivot_longer(
        cols = all_of(pathology_vars),
        names_to = "pathology",
        values_to = "score"
      ) %>%
      filter(!is.na(score)) %>%
      mutate(
        score = as.factor(score),
        pathology = factor(pathology,
                           levels = pathology_vars,
                           labels = c("Mesangial hypercellularity",
                                      "Endocapillary hypercellularity",
                                      "Segmental sclerosis",
                                      "Tubular atrophy/Interstitial fibrosis",
                                      "Crescents"))
      )

    # 计算病理特征阳性率
    pathology_percent <- pathology_data %>%
      group_by(qwen_pca, rapid_up24, pathology, score) %>%
      summarise(count = n(), .groups = 'drop') %>%
      group_by(qwen_pca, rapid_up24, pathology) %>%
      mutate(
        total = sum(count),
        percentage = count / total * 100
      )


    # 2. nature标准颜色方案（针对MEST-C特征）
    # 使用色盲友好的调色板
    mestc_colors <- c(
      "#1A85FF",  # 蓝色 - M1
      "#D41159",  # 红色 - E1
      "#40B0A6",  # 青色 - S1
      "#E1BE6A",  # 金色 - C1
      "#994F00",  # 棕色 - T1
      "#005AB5"   # 深蓝 - T2
    )[1:length(unique(pathology_percent$score))]

    # 3. 改进的主题函数（针对病理图）
    theme_nature_pathology <- function(base_size = 10) {
      theme_minimal(base_family = "Arial", base_size = base_size) %+replace%
        theme(
          plot.title = element_text(face = "bold", hjust = 0.5,
                                    size = base_size + 1, margin = margin(b = 10)),
          plot.subtitle = element_text(hjust = 0.5, color = "gray40",
                                       size = base_size, margin = margin(b = 8)),
          axis.title = element_text(face = "bold", size = base_size),
          axis.title.x = element_text(margin = margin(t = 8)),
          axis.title.y = element_text(margin = margin(r = 8)),
          axis.text = element_text(color = "black", size = base_size - 1),
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          legend.title = element_text(face = "bold", size = base_size),
          legend.text = element_text(size = base_size - 1),
          legend.position = "right",
          legend.key.size = unit(0.5, "cm"),
          strip.text = element_text(face = "bold", size = base_size,
                                    margin = margin(b = 4)),
          strip.background = element_rect(fill = "grey95", color = NA),
          panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
          panel.grid.minor = element_blank(),
          panel.spacing = unit(0.6, "lines"),
          plot.margin = margin(10, 10, 10, 10),
          plot.background = element_rect(fill = "white", color = NA)
        )
    }

    # 4. 改进的病理特征图
    plot3_improved <- pathology_percent %>%
      ggplot(aes(x = qwen_pca, y = percentage,
                 fill = score, group = score)) +
      # 使用geom_col代替geom_bar(stat="identity")
      geom_col(position = position_dodge(width = 0.75),
               width = 0.65,
               color = "white",
               linewidth = 0.3) +
      # 添加数值标签（更精细的控制）
      geom_text(
        aes(label = ifelse(percentage > 5,
                           sprintf("%.0f%%", percentage),
                           sprintf("%.1f%%", percentage))),
        position = position_dodge(width = 0.75),
        vjust = -0.4,
        size = 2.8,
        family = "Arial",
        fontface = "bold"
      ) +
      # 分面布局 - 病理特征在行，风险组在列
      facet_grid(
        pathology ~ rapid_up24,
        scales = "free_x",
        space = "free_x",
        switch = "y" # 将分面标签移到左侧
        # labeller = labeller(
        #   rapid_up24 = c(
        #     "Standard-risk" = "Standard Risk",
        #     "High-risk" = "High Risk"
        #   )
        # )
      ) +
      # 颜色方案
      scale_fill_manual(
        values = mestc_colors,
        name = "MEST-C Score",
        guide = guide_legend(nrow = 2, byrow = TRUE)
      ) +
      # Y轴设置 - 移除标签和数值
      scale_y_continuous(
        limits = c(0, max(pathology_percent$percentage) * 1.25),
        expand = expansion(mult = c(0, 0.1)),
        breaks = scales::pretty_breaks(n = 6),
        labels = NULL  # 移除Y轴数值标签
      ) +
      # 坐标轴标签
      labs(
        x = NULL,
        y = NULL,  # 移除Y轴标签
        # title = "MEST-C Pathological Features by Cluster and Risk Group",
        # subtitle = "Percentage of patients with each pathological feature",
        # caption = "MEST-C: Oxford classification of IgA nephropathy\nM: Mesangial hypercellularity, E: Endocapillary hypercellularity,\nS: Segmental sclerosis, T: Tubular atrophy/interstitial fibrosis, C: Crescents"
      ) +
      # 应用主题
      theme_nature_pathology(base_size = 10) +
      # 额外的主题调整
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.text.y = element_blank(),  # 移除Y轴数值
        axis.ticks.y = element_blank(),  # 移除Y轴刻度
        axis.title.y = element_blank(),  # 确保Y轴标题被移除
        strip.text.y.left = element_text(angle = 0, hjust = 1, size = 9, margin = margin(r = 10)),  # 左侧分面标签
        strip.placement = "outside",  # 将分面标签放在图形外部
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.margin = margin(t = 0, b = 5),
        panel.grid.major.y = element_line(color = "grey92", linewidth = 0.2),
        panel.grid.major.x = element_blank(),
        panel.spacing = unit(0.8, "lines")  # 增加面板间距
      )

    ### 保存病理特征图
    CairoPDF("results/Figure3_Pathological_Features_nature.pdf",
             width = 10,
             height = 12,
             family = "Arial")
    print(plot3_improved)
    dev.off()

  }

  ## save plots data
  if(FALSE){
    save(clinical_interaction_data, pathology_percent, pathology_data,
    file = "dataPlots/Figure_Data_Clinical_Pathology_nature.RData")
  }


  # 7. 创建桑基图风格的数据流图（展示聚类和风险组的关系）
  # 计算转移矩阵
  flow_data <- df %>%
    group_by(qwen_pca, rapid_up24) %>%
    summarise(count = n(), .groups = 'drop') %>%
    group_by(qwen_pca) %>%
    mutate(
      cluster_total = sum(count),
      cluster_percent = count / cluster_total * 100
    ) %>%
    group_by(rapid_up24) %>%
    mutate(
      risk_total = sum(count),
      risk_percent = count / risk_total * 100
    )

  # 图6: 聚类与风险组的关系

  plot6b <- flow_data %>%
    ggplot(aes(x = qwen_pca, y = count, fill = rapid_up24)) +
    geom_bar(stat = "identity", position = "stack") +
    geom_text(aes(label = paste(count, "\n", sprintf("%.0f%%", cluster_percent))),
              position = position_stack(vjust = 0.5),
              size = 3, family = "Arial") +
    scale_fill_brewer(palette = "Set1", name = "Risk Group") +
    labs(x = "Cluster", y = "Number of Patients",
         title = "Risk Group Distribution within Clusters") +
    theme_arial_publication(base_size = 11) +
    theme(
      legend.position = "right"
    )

  plot6c <- flow_data %>%
    ggplot(aes(x = rapid_up24, y = count, fill = qwen_pca)) +
    geom_bar(stat = "identity", position = "stack") +
    geom_text(aes(label = paste(count, "\n", sprintf("%.0f%%", risk_percent))),
              position = position_stack(vjust = 0.5),
              size = 3, family = "Arial") +
    scale_fill_brewer(palette = "Set2", name = "Cluster") +
    labs(x = "Risk Group", y = "Number of Patients",
         title = "Cluster Distribution within Risk Groups") +
    theme_arial_publication(base_size = 11) +
    theme(
      legend.position = "right"
    )


}

#### baseline Table FOR rapid_up24
if (FALSE) {
  demographic_vars <- c("age", "Gender", "systolic_bp", "diastolic_bp")


  diagnosis_vars <- c(
    "hypertentsion", "diabetic", "qwen_tsne"
  )

  # 实验室指标（选择关键指标）
  lab_vars <- c(
    "s_alb", "s_cr", "s_ua", "s_hb", "up24", "eGFR_onset", "eGFR_category_onset"
  )

  # 病理指标
  pathology_vars <- c(
    "MEST_C_score_M", "MEST_C_score_E", "MEST_C_score_S",
    "MEST_C_score_T", "MEST_C_score_C"
  )

  # 治疗药物（选择主要治疗药物）
  medication_vars <- c(
    "steroid", "rasb", "mmf", "ctx", "fk506", "aza", 'leflunomide', 'immuno'
  )

  # 合并所有变量
  all_vars <- c(demographic_vars, diagnosis_vars, lab_vars, pathology_vars, medication_vars)

  # 定义分类变量
  categorical_vars <- c(
    "Gender", "eGFR_category_onset", "MEST_C_score_M", "MEST_C_score_E", "MEST_C_score_S",
    "MEST_C_score_T", "MEST_C_score_C", "steroid", "rasb", "mmf", "ctx", "fk506", "aza",
    "hypertentsion", "diabetic", "qwen_tsne", 'leflunomide', 'immuno'
  )


  # 创建TableOne对象
  table_one <- CreateTableOne(
    vars = all_vars,
    data = df,
    factorVars = categorical_vars
  )

  # 打印表格
  table_one_print <- print(
    table_one,
    nonnormal = c("s_cr", "eGFR_onset", "up24"),
    exact = categorical_vars,
    quote = FALSE,
    noSpaces = TRUE,
    printToggle = FALSE,
    showAllLevels = FALSE,
    contDigits = 1,  # 连续变量小数位数
    catDigits = 1,   # 分类变量小数位数
    pDigits = 3      # p值小数位数
  )

  convert_row_names <- function(table_df) {
    rownames_vec <- rownames(table_df)
    new_rownames <- character(length(rownames_vec))

    for (i in seq_along(rownames_vec)) {
      rname <- rownames_vec[i]

      if (rname == "n") {
        new_rownames[i] <- "n"
      } else if (rname == "age (mean (SD))") {
        new_rownames[i] <- "Age (years)"
      } else if (rname == "Gender = 女 (%)") {
        new_rownames[i] <- "Female"
      } else if (rname == "systolic_bp (mean (SD))") {
        new_rownames[i] <- "Systolic blood pressure (mmHg)"
      } else if (rname == "diastolic_bp (mean (SD))") {
        new_rownames[i] <- "Diastolic blood pressure (mmHg)"
      } else if (rname == "hypertentsion = 1 (%)") {
        new_rownames[i] <- "Hypertension"
      } else if (rname == "diabetic = 1 (%)") {
        new_rownames[i] <- "Diabetes"
      } else if (rname == "qwen_tsne = Cluster 1 (%)") {
        new_rownames[i] <- "Cluster 1"
      } else if (rname == "s_alb (mean (SD))") {
        new_rownames[i] <- "Serum albumin (g/L)"
      } else if (rname == "s_cr (mean (SD))" | rname == "s_cr (median [IQR])") {
        new_rownames[i] <- "Serum creatinine (μmol/L)"
      } else if (rname == "s_ua (mean (SD))") {
        new_rownames[i] <- "Serum uric acid (μmol/L)"
      } else if (rname == "s_hb (mean (SD))") {
        new_rownames[i] <- "Hemoglobin (g/L)"
      } else if (rname == "up24 (mean (SD))" | rname == "up24 (median [IQR])") {
        new_rownames[i] <- "24-hour urine protein (g/24h)"
      } else if (rname == "eGFR_onset (mean (SD))" | rname == "eGFR_onset (median [IQR])") {
        new_rownames[i] <- "eGFR (ml/min/1.73m²)"
      } else if (rname == "eGFR_category_onset (%)") {
        new_rownames[i] <- "eGFR category"
      } else if (grepl("^   G", rname)) {
        # 这些是eGFR分级的子项，保持原样在后面处理
        new_rownames[i] <- rname
      } else if (rname == "s_iga (mean (SD))") {
        new_rownames[i] <- "Serum IgA (g/L)"
      } else if (rname == "s_igg (mean (SD))") {
        new_rownames[i] <- "Serum IgG (g/L)"
      } else if (rname == "s_igm (mean (SD))") {
        new_rownames[i] <- "Serum IgM (g/L)"
      } else if (rname == "MEST_C_score_M = 1 (%)") {
        new_rownames[i] <- "MEST-C M1"
      } else if (rname == "MEST_C_score_E = 1 (%)") {
        new_rownames[i] <- "MEST-C E1"
      } else if (rname == "MEST_C_score_S = 1 (%)") {
        new_rownames[i] <- "MEST-C S1"
      } else if (rname == "MEST_C_score_T (%)") {
        new_rownames[i] <- "MEST-C T score"
      } else if (grepl("^   [0-2]", rname)) {
        # 这些是T分期的子项
        new_rownames[i] <- rname
      } else if (rname == "MEST_C_score_C = 1 (%)") {
        new_rownames[i] <- "MEST-C C1"
      } else if (rname == "steroid = 1 (%)") {
        new_rownames[i] <- "Corticosteroid therapy"
      } else if (rname == "rasb = 1 (%)") {
        new_rownames[i] <- "RAS blockade therapy"
      } else if (rname == "mmf = 1 (%)") {
        new_rownames[i] <- "Mycophenolate mofetil"
      } else if (rname == "ctx = 1 (%)") {
        new_rownames[i] <- "Cyclophosphamide"
      } else if (rname == "fk506 = 1 (%)") {
        new_rownames[i] <- "Tacrolimus"
      } else if (rname == "aza = 1 (%)") {
        new_rownames[i] <- "Azathioprine"
      } else if (rname == "leflunomide = 1 (%)") {
        new_rownames[i] <- "Leflunomide"
      } else if (rname == "immuno = Yes (%)") {
        new_rownames[i] <- "Immunosuppressive therapy"
      } else {
        new_rownames[i] <- rname
      }
    }
    return(new_rownames)
  }

  # 应用行名转换
  new_rownames <- convert_row_names(table_one_print)
  table_df <- as.data.frame(table_one_print)
  rownames(table_df) <- new_rownames

  # 创建TableOne对象
  table_one_qwen <- CreateTableOne(
    vars = all_vars,
    data = df,
    factorVars = categorical_vars,
    strata = 'rapid_up24'
  )

  # 打印表格
  table_qwen_print <- print(
    table_one_qwen,
    nonnormal = c("s_cr", "eGFR_onset", "up24"),
    quote = FALSE,
    noSpaces = TRUE,
    printToggle = FALSE,
    showAllLevels = FALSE,
    contDigits = 1,  # 连续变量小数位数
    catDigits = 1,   # 分类变量小数位数
    pDigits = 3      # p值小数位数
  )

  # 应用行名转换
  new_rownames <- convert_row_names(table_qwen_print)
  table_df_qwen <- as.data.frame(table_qwen_print)
  rownames(table_df_qwen) <- new_rownames
  table_full_qwen <- cbind(table_df, table_df_qwen) %>%
    select(-'test') %>%
    rename("p value" = "p")
  # 创建flextable对象 - 三线表格式
  # 计算列数（如果尚未计算）
  ncols <- ncol(table_full_qwen)

  flex_table <- table_full_qwen %>%
    tibble::rownames_to_column(var = "Characteristics") %>%
    flextable() %>%
    fontsize(size = 8, part = "all") %>%
    flextable::font(fontname = "Times New Roman", part = "all") %>%
    border_remove() %>%
    hline_top(border = fp_border(width = 1.5), part = "header") %>%
    hline(i = 1, border = fp_border(width = 1.5), part = "header") %>%
    hline_bottom(border = fp_border(width = 1.5), part = "body") %>%
    align(align = "center", part = "header") %>%
    align(align = "center", part = "body") %>%
    align(j = 1, align = "left", part = "body") %>%
    bold(i = 1, bold = TRUE, part = "header") %>%
    line_spacing(space = 0.7, part = "body") %>%
    line_spacing(space = 0.7, part = "header") %>%
    autofit() %>%                              # 先自适应，再固定宽度
    set_table_properties(layout = "fixed") %>% # 固定布局，便于精确控制宽度
    width(j = 1, width = 1.5) %>%              # 第一列较宽（特征名）
    width(j = 2:ncols, width = 1.0)            # 其它列宽减小以缩小间距


  # 创建word文档
  doc <- read_docx() %>%
    # 添加表格标题
    body_add_par("sup Table ?: Baseline Characteristics of IgA Nephropathy Patients by clinical high-risk classification",
                 style = "table title") %>%
    # 添加表格
    body_add_flextable(flex_table) %>%
    # 添加注释
    body_add_par("Note: Continuous variables are presented as mean (SD) or median [IQR]; categorical variables are presented as frequency (%). eGFR, estimated glomerular filtration rate; MEST-C, Oxford classification of IgA nephropathy; RAS, renin-angiotensin system; Immunosuppressive therapy was defined as initial administration of any immunosuppressive agent - including mycophenolate mofetil, cyclophosphamide, tacrolimus, azathioprine, or leflunomide - within one year following renal biopsy.",
                 style = "Normal") %>%
    # 设置页面布局
    body_end_section_continuous()  # 连续分节，确保表格不跨页

  # 保存文档
  print(doc, target = "results/IgAN_Baseline_Table_clusters_rapid_up24.docx")
}

### rapid_up24
if (FALSE) {

  rapid_table <- df_features %>%
    select("keyID", "bioID") %>%
    distinct() %>%
    left_join(
      df %>% select("keyID", "rapid_up24")
    ) %>%
    filter(!is.na(rapid_up24)) %>%
    group_by(keyID) %>%
    slice_min(
      order_by = bioID,
      n = 1,
      with_ties = FALSE
    ) %>%
    rename('bio_id' = bioID)

  write.csv(
    rapid_table,
    file = "data/IgAN_rapid_up24_labels.csv",
    row.names = FALSE
  )

}
