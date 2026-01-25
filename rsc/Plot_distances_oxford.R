#######
rm(list = ls())
library(ggsignif)
library(effectsize)
library(dplyr)
library(tidyr)
library(ggplot2)
library(Cairo)
library(ggtext)
library(jsonlite)
library(ggridges)
library(extrafont)
library(rstatix)
library(RColorBrewer)
library(irr)

loadfonts(device = "win")  
par(family = "Arial")
### 数据准备 2025-12-22, 改deepseek
if (TRUE) {
  load("data/statistics_data_plus_2025-12-14.RData")

  par(family = "Arial")

  df <- df_onset %>% #  
    filter(rowSums(select(., starts_with("malignancy"))) == 0) %>%    # 124
    mutate(
      up24 = ifelse(up24 > 100, up24 / 1000, up24)
    ) %>%
    rename(
      deepseek_pca = kmeans_pca_deepseek,
    ) %>%
    mutate(
      deepseek_pca = factor(deepseek_pca, levels = c(0, 1),
                            labels = c("low-activity", "high-activity")),
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

}

######## 真实的MEST-C分布
if (TRUE) {
  df <- df %>%
    mutate(
      mestc = paste0("M.", MEST_C_score_M, ".",
                     "E.", MEST_C_score_E, ".",
                     "S.", MEST_C_score_S, ".",
                     "T.", MEST_C_score_T, ".",
                     "C.", MEST_C_score_C)
    )
  
  ############ up24 plot
  if (TRUE) {
    ### mestc distribution, ggpl
    mestc_dist <- df %>%
      mutate(
        rapid_up24 = ifelse(rapid_up24 == "Yes", "high-risk", "low-risk")
      ) %>%
      filter(!is.na(rapid_up24)) %>%
      # 首先过滤掉NA值
      filter(!(is.na(MEST_C_score_C) |
        is.na(MEST_C_score_E) |
        is.na(MEST_C_score_M) |
        is.na(MEST_C_score_S) |
        is.na(MEST_C_score_T))) %>%
      # 计算总数用于后续比例比较
      group_by(mestc) %>%
      mutate(total_n = n()) %>%
      ungroup() %>%
      # 按cluster分组统计
      count(mestc, rapid_up24, total_n) %>%  # 注意：这里改成了rapid_up24
      # 确保每个mestc都有所有可能的cluster值（用0填充缺失的）
      complete(mestc, rapid_up24,
               fill = list(n = 0)) %>%
      # 重新获取total_n的值（因为complete可能创建了缺失total_n的行）
      group_by(mestc) %>%
      mutate(total_n = ifelse(is.na(total_n),
                              first(na.omit(total_n)),
                              total_n)) %>%
      ungroup() %>%
      # 确保total_n没有NA值
      mutate(total_n = ifelse(is.na(total_n), 0, total_n)) %>%
      # 计算每个cluster内的百分比
      group_by(rapid_up24) %>%
      mutate(
        pct_within_cluster = n / sum(n) * 100,
        pct_label = sprintf("%.1f%%", pct_within_cluster)
      ) %>%
      ungroup() %>%
      # 计算总体中的百分比
      mutate(
        pct_overall = n / total_n * 100
      ) %>%
      # 计算不同cluster间的比例差异
      group_by(mestc) %>%
      mutate(
        pct_diff = pct_within_cluster[rapid_up24 == "low-risk"] -
          pct_within_cluster[rapid_up24 == "high-risk"],
        abs_pct_diff = abs(pct_diff),
        # 添加cluster名称
        cluster_name = as.character(rapid_up24)
      ) %>%
      ungroup() %>%
      # 按差异排序
      arrange(desc(abs_pct_diff)) %>%
      mutate(
        mestc = factor(mestc, levels = unique(mestc)),
        cluster_label = cluster_name,
        diff_label = ifelse(abs_pct_diff > 0,
                            sprintf("Δ=%.1f%%", abs_pct_diff),
                            "")
      )

    mestc_dist_up <- mestc_dist %>%
      mutate(group = "up")

    # 准备条形图数据
    bar_data <- mestc_dist_up %>%
      mutate(mestc = factor(mestc, levels = mestc_levels))  # 重新排序因子

    # 准备点线图数据
    line_data <- bar_data %>%
      select(mestc, pct_within_cluster, rapid_up24, cluster_name) %>%
      pivot_wider(
        id_cols = mestc,
        names_from = cluster_name,
        values_from = pct_within_cluster
      ) %>%
      mutate(
        diff = `low-risk` - `high-risk`,
        abs_diff = abs(diff),
        # 确保使用相同的因子顺序
        mestc = factor(mestc, levels = mestc_levels)
      )

    # 创建条形图
    g_bars <- ggplot(bar_data, aes(x = n, y = mestc)) +
      geom_col(
        aes(fill = cluster_name),
        width = 0.6,
        color = "white",
        linewidth = 0.2,
        position = position_dodge(width = 0.7)
      ) +
      # 使用预设颜色
      scale_fill_manual(
        values = c("low-risk" = "#0571b0", "high-risk" = "#ca0020"),
        name = "Clinical Risk Stratification"
      ) +
      scale_y_discrete(
        limits = rev(mestc_levels)  # 反转顺序以确保顶部是最大差异
      ) +
      scale_x_continuous(
        breaks = scales::pretty_breaks(n = 5),
        labels = scales::comma_format()
      ) +
      labs(
        x = "Number of Patients",
        y = "Oxford Classification Patterns"
      ) +
      theme_minimal(base_size = 9, base_family = "Arial") +
      theme(
        text = element_text(color = "black"),
        axis.line = element_line(color = "black", linewidth = 0.3),
        axis.ticks = element_line(color = "black", linewidth = 0.3),
        axis.ticks.length = unit(1.5, "pt"),
        axis.text = element_text(color = "black", size = 8),
        axis.text.y = element_text(
          margin = margin(r = 3),
          face = "bold",
          color = "black"
        ),
        axis.title = element_text(face = "bold", size = 9, color = "black"),
        axis.title.x = element_text(margin = margin(t = 8)),
        axis.title.y = element_text(margin = margin(r = 8)),
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.2),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey95", linewidth = 0.1),
        panel.grid.minor.y = element_blank(),
        plot.margin = margin(1, 0, 3, 1),
        legend.position = c(0.8, 0.2),
        legend.title = element_text(face = "bold", size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.5, "cm"),
        legend.margin = margin(b = 5)
      )

    # 创建点线图
    g_lines <- ggplot(line_data, aes(x = diff, y = mestc)) +
      # 添加垂直线表示差异
      geom_segment(
        aes(x = 0, xend = diff, y = mestc, yend = mestc),
        color = "gray70",
        linewidth = 0.5,
        linetype = "dashed"
      ) +
      geom_point(
        aes(x = diff,
            shape = "Cluster 1",
            color = ifelse(diff > 0, "#0571b0", "#ca0020"),  # 根据diff正负设置颜色
            fill = ifelse(diff > 0, "#0571b0", "#ca0020")),  # 根据diff正负设置填充
        size = 2.5
      ) +
      # 连接两个点的线，根据diff正负设置颜色
      geom_segment(
        aes(x = 0, xend = diff, y = mestc, yend = mestc),
        color = 'grey50',  # 线也根据diff正负变色
        linewidth = 0.1
      ) +
      # 添加垂直的0线
      geom_vline(
        xintercept = 0,
        color = "black",
        linewidth = 0.3,
        linetype = "solid"
      ) +
      scale_y_discrete(
        limits = rev(mestc_levels),  # 使用相同的顺序
        position = "right"  # Y轴放在右侧
      ) +
      scale_x_continuous(
        limits = c(
          min(line_data$diff, 0) * 1.01,
          max(line_data$diff, 0) * 1.01
        ),
        breaks = scales::pretty_breaks(n = 4),
        labels = function(x) paste0(sprintf("%+.1f", x), "%")
      ) +
      scale_shape_manual(
        values = c("Cluster 0" = 21, "Cluster 1" = 19),
        name = "Cluster Points"
      ) +
      scale_color_identity() +  # 使用实际颜色值
      scale_fill_identity() +   # 使用实际填充值
      labs(
        x = "Percentage Difference",
        y = NULL
      ) +
      theme_minimal(base_size = 9, base_family = "Arial") +
      theme(
        text = element_text(color = "black"),
        axis.line = element_line(color = "black", linewidth = 0.3),
        axis.ticks = element_line(color = "black", linewidth = 0.3),
        axis.ticks.length = unit(1.5, "pt"),
        axis.text = element_text(color = "black", size = 8),
        axis.text.y = element_blank(),
        axis.title = element_text(face = "bold", size = 9, color = "black"),
        axis.title.x = element_text(margin = margin(t = 8)),
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.2),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey95", linewidth = 0.1),
        panel.grid.minor.y = element_blank(),
        plot.margin = margin(1, 1, 3, 0),
        legend.position = "none"
      )

    combined_plot_up <- cowplot::plot_grid(
      g_bars, g_lines,
      ncol = 2,
      align = "h",
      labels = c("B", ""),
      rel_widths = c(1, 0.6)
    )


    # 首先设置系统字体（使用Windows系统字体）
    if (.Platform$OS.type == "windows") {
      windowsFonts(Arial = windowsFont("Arial"))
    }


    # 保存为PDF
    CairoPDF("results/mestc_distribution_rapid_up_plus.pdf",
             width = 8, height = 10)
    combined_plot_up
    dev.off()

  }
  

  combined_plot <- cowplot::plot_grid(
    combined_plot_llm, combined_plot_up,
    ncol = 2,
    align = "v",
    labels = c("", ""),
    rel_heights = c(1, 1)
  )
  

  # 保存为PDF
  CairoPDF("results/mestc_distribution_combined_llm_up.pdf",
           width = 16, height = 10)
  combined_plot
  dev.off()
  # data prepare
  if (TRUE) {
    # 合并LLM和UP数据，统一处理
    mestc_dist <- bind_rows(
      llm = mestc_dist_llm,
      up = mestc_dist_up,
      .id = "group"  # 保留来源信息用于分类
    ) %>%
      # 创建更清晰的分类标签
      mutate(
        classification = case_when(
          group == "llm" ~ "Pathology Subtypes",
          group == "up" ~ "Clinical Risk Stratification",
          TRUE ~ NA_character_
        )
      ) %>%
      select(-group) %>%  # 移除原始分组变量

      # 按分类和mestc计算平均百分比差异
      group_by(classification, mestc) %>%
      summarise(
        pct_diff = ifelse(sum(pct_diff) > 0, "positive", "negative"),
        .groups = "drop"  # 自动取消分组，避免后续重复操作
      ) %>%

      # 计算每个mestc内两组分类的差异符号一致性
      group_by(mestc) %>%
      mutate(
        # 检查两个分类的pct_diff是否同号
        diff_sign = if_else(length(unique(pct_diff)) == 1, "positive", "negative"),
      ) %>%
      ungroup() %>%
      select(mestc, diff_sign, classification, pct_diff) %>%
      distinct()  # 去重，确保每个mestc只有一行

    # 可选：对结果进行排序，提高可读性
    mestc_dist_first_hospital <- mestc_dist %>%
      arrange(mestc)


    # 将数据转换为每个观察者-观察对象的矩阵
    wide_data <- mestc_dist_first_hospital %>%
      select(mestc, classification, pct_diff) %>%
      pivot_wider(
        names_from = classification,  # 观察者作为列
        values_from = pct_diff,       # 观察结果作为值
        names_prefix = "Rater_"
      )

    # print(head(wide_data))

    library(irr)

    # 转换数据格式
    kappa_data <- as.matrix(wide_data[, -1])  # 移除mestc列

    # 计算Fleiss' Kappa（适用于多个评分者）
    # 首先确保所有评分者对每个对象都有评分
    kappa_result <- kappam.fleiss(kappa_data)
    print(kappa_result)


    save(mestc_dist_first_hospital, pairwise_results,
         file = "data/mestc_distance_difference_sign_first_hospital.Rdata")


  }

  #### logistic regression
  if (TRUE) {
    # Load required packages
    library(tidyverse)
    library(gtsummary)
    library(flextable)
    library(dplyr)


    # Fit two logistic regression models
    fit_llm <- glm(
      formula = as.factor(deepseek_pca) ~ as.factor(MEST_C_score_M) +
        as.factor(MEST_C_score_E) +
        as.factor(MEST_C_score_S) +
        as.factor(MEST_C_score_T) +
        as.factor(MEST_C_score_C),
      data = df,
      family = binomial(link = "logit")
    )

    fit_up <- glm(
      formula = rapid_up24 ~ as.factor(MEST_C_score_M) +
        as.factor(MEST_C_score_E) +
        as.factor(MEST_C_score_S) +
        as.factor(MEST_C_score_T) +
        as.factor(MEST_C_score_C),
      data = df,
      family = binomial(link = "logit")
    )

    # Create a custom function to format P values
    format_p_value <- function(x) {
      # Format P values with significance stars
      sapply(x, function(p) {
        if (is.na(p)) return("")
        if (p < 0.001) {
          return("<0.001***")
        } else if (p < 0.01) {
          return(sprintf("%.3f**", p))
        } else if (p < 0.05) {
          return(sprintf("%.3f*", p))
        } else {
          return(sprintf("%.3f", p))
        }
      })
    }

    # Create tables for each model (使用正确的 gtsummary 语法)
    tbl_llm <- tbl_regression(
      fit_llm,
      exponentiate = TRUE,
      label = list(
        "as.factor(MEST_C_score_M)" = "M Score",
        "as.factor(MEST_C_score_E)" = "E Score",
        "as.factor(MEST_C_score_S)" = "S Score",
        "as.factor(MEST_C_score_T)" = "T Score",
        "as.factor(MEST_C_score_C)" = "C Score"
      ),
      estimate_fun = function(x) style_number(x, digits = 2)
    ) %>%
      modify_column_unhide(columns = c(std.error, p.value)) %>%
      # 使用正确的语法格式化 P 值
      modify_fmt_fun(
        update = list(p.value = format_p_value)  # 保留 update 参数避免警告
      ) %>%
      modify_header(
        label = "**Variable**",
        estimate = "**OR (95% CI)**",
        std.error = "**SE**",
        p.value = "***P* Value**"  # Italic and bold header
      )

    tbl_up <- tbl_regression(
      fit_up,
      exponentiate = TRUE,
      label = list(
        "as.factor(MEST_C_score_M)" = "M Score",
        "as.factor(MEST_C_score_E)" = "E Score",
        "as.factor(MEST_C_score_S)" = "S Score",
        "as.factor(MEST_C_score_T)" = "T Score",
        "as.factor(MEST_C_score_C)" = "C Score"
      ),
      estimate_fun = function(x) style_number(x, digits = 2)
    ) %>%
      modify_column_unhide(columns = c(std.error, p.value)) %>%
      # 使用正确的语法格式化 P 值
      modify_fmt_fun(
        update = list(p.value = format_p_value)  # 保留 update 参数避免警告
      ) %>%
      modify_header(
        label = "**Variable**",
        estimate = "**OR (95% CI)**",
        std.error = "**SE**",
        p.value = "***P* Value**"  # Italic and bold header
      )

    # Merge two tables
    final_table <- tbl_merge(
      list(tbl_llm, tbl_up),
      tab_spanner = c("**Pathology Classification**", "**Clinical Risk Stratification**")
    )
    
    # 首先转换为 flextable
    nature_flex <- final_table %>%
      as_flex_table()

    # 应用三线表格式 - 使用正确的边框设置方法
    # 创建边框对象
    thick_border <- officer::fp_border(color = "black", width = 1.5)
    thin_border <- officer::fp_border(color = "black", width = 0.75)

    # 应用格式化
    nature_flex <- nature_flex %>%
      # Set font to Arial, size 10
      flextable::font(fontname = "Arial", part = "all") %>%
      flextable::fontsize(size = 10, part = "all") %>%
      # Bold header
      bold(part = "header") %>%
      # Make P value column italic (column indices: 1=label, 2=OR1, 3=SE1, 4=P1, 5=OR2, 6=SE2, 7=P2)
      # For Model 1 P values (column 4)
      italic(j = 4, part = "body") %>%
      # For Model 2 P values (column 7)
      italic(j = 7, part = "body") %>%
      # Center alignment for header and body
      align(align = "center", part = "header") %>%
      align(align = "center", part = "body") %>%
      # Left align variable names
      align(j = 1, align = "left", part = "body") %>%
      # Set paragraph spacing: before=0, after=0, line spacing=1
      line_spacing(space = 1, part = "all") %>%
      padding(padding = 0, part = "all") %>%
      # 移除所有边框
      border_remove()

    # 添加三线表边框 - 使用正确的方法
    # 顶部边框
    nature_flex <- nature_flex %>%
      hline_top(border = thick_border, part = "all")

    # 底部边框
    nature_flex <- nature_flex %>%
      hline_bottom(border = thick_border, part = "all")

    # 表头底部边框（细线）
    nature_flex <- nature_flex %>%
      hline(i = 1, border = thin_border, part = "header")

    # 继续其他格式化
    nature_flex <- nature_flex %>%
      # Set table properties
      set_table_properties(layout = "autofit", width = 1) %>%
      # Add table title
      add_header_lines(values = "Table 1. Association of MEST-C Histopathological Scores with Pathology Classification and Clinical Risk Stratification") %>%
      flextable::font(fontname = "Arial", part = "header", i = 1) %>%
      flextable::fontsize(size = 10, part = "header", i = 1) %>%
      bold(part = "header", i = 1) %>%
      align(align = "center", part = "header", i = 1) %>%
      # Add spacing after title
      padding(padding.top = 6, part = "header", i = 1) %>%
      padding(padding.bottom = 6, part = "header", i = 1)

    # Add detailed footnotes
    nature_flex <- nature_flex %>%
      add_footer_lines(
        values = c(
          "Abbreviations: MEST-C, Oxford classification of IgA nephropathy; M, mesangial hypercellularity; E, endocapillary hypercellularity;",
          "S, segmental sclerosis; T, tubular atrophy/interstitial fibrosis; C, crescents; OR, odds ratio; CI, confidence interval.",
          "Note: All variables are categorical with score 0 as the reference category. T score has three categories: T0 (reference), T1, and T2.",
          "Analyses were performed using multivariable logistic regression with adjustment for all MEST-C components simultaneously.",
          "Significance codes: ***P < 0.001, **P < 0.01, *P < 0.05."
        )
      ) %>%
      # Format footer
      flextable::font(fontname = "Arial", part = "footer") %>%
      flextable::fontsize(size = 9, part = "footer") %>%
      line_spacing(space = 1, part = "footer") %>%
      padding(padding = 2, part = "footer") %>%
      align(align = "left", part = "footer")

    # Display the table
    print(nature_flex)

    # Save as Word document
    save_as_docx(
      nature_flex,
      path = "results/Table1_MESTC_Logistic_Regression.docx",
      pr_section = prop_section(
        page_size = page_size(orient = "portrait", width = 8.3, height = 11.7),
        type = "continuous",
        page_margins = page_mar(bottom = 1, top = 1, right = 1, left = 1)
      )
    )

  }


}
##### oxford-c distribution
if (TRUE) {
  oxford_dist <- read.csv(
    'data\\oxford_feature_distances_qwen3_plus.csv',
    stringsAsFactors = FALSE,
    colClasses = "character"
  ) %>%
    left_join(
      pathology_features_processed %>%
        select("keyID", "bio_id") %>%
        distinct(), by = c("bio_id" = "bio_id")
    ) %>%
    group_by(keyID) %>%
    slice_min(
      order_by = bio_id,
      n = 1,
      with_ties = FALSE
    ) %>%
    pivot_longer(
      cols = starts_with("M"),
      names_to = "oxford_c",
      values_to = "distance_value") %>%
    mutate(
      kmeans_cluster = factor(kmeans_cluster, levels = c(0, 1), # 注意计算距离时一个更换顺序
                              labels = c("low-activity", "high-activity")),
      distance_value = as.numeric(distance_value)
    ) %>%
    mutate(
      oxford_c = factor(
        oxford_c,
        levels = mestc_levels
      )) %>%
    mutate(
      classification = "Pathology Subtype"
    ) %>%
    bind_rows(
      read.csv(
        'data\\oxford_feature_distances_qwen3_plus.csv',
        stringsAsFactors = FALSE,
        colClasses = "character"
      ) %>%
        select(-kmeans_cluster) %>%
        left_join(
          pathology_features_processed %>%
            select("keyID", "bio_id") %>%
            distinct(), by = c("bio_id" = "bio_id")
        ) %>%
        group_by(keyID) %>%
        slice_min(
          order_by = bio_id,
          n = 1,
          with_ties = FALSE
        ) %>%
        ungroup() %>%
        left_join(
          df %>%
            select(keyID, rapid_up24) %>%
            filter(!is.na(rapid_up24)),
          by = "keyID") %>%
        mutate(
          kmeans_cluster = factor(rapid_up24, levels = c("No", "Yes"),
                                  labels = c("low-risk", "high-risk")) # 注意为了代码不用大改
        ) %>%
        select(-rapid_up24) %>%
        pivot_longer(
          cols = starts_with("M"),
          names_to = "oxford_c",
          values_to = "distance_value") %>%
        mutate(
          distance_value = as.numeric(distance_value)
        ) %>%
        ### select real type
        mutate(
          oxford_c = factor(
            oxford_c,
            levels = mestc_levels
          )) %>%
        mutate(
          classification = "Clinical Risk Stratification"
        )) %>%
    # mutate(
    #   distance_value = scale(distance_value)[, 1]
    # ) %>%
    mutate(
      kmeans_cluster = factor(kmeans_cluster, levels = c("low-activity", "high-activity",
                                                         "low-risk", "high-risk"))
    ) %>%
    mutate(
      classification = factor(classification,
                              levels = c("Pathology Subtype",
                                         "Clinical Risk Stratification"))
    ) %>%
    na.omit()


  # oxford_dist_cluster
  if (TRUE) {
    library(ggsignif)
    library(effectsize)
    library(dplyr)
    library(ggplot2)
    library(ggridges)

    # 使用rstatix进行成对t检验
    pairwise_results <- oxford_dist %>%
      group_by(oxford_c, classification) %>%
      pairwise_t_test(
        distance_value ~ kmeans_cluster,
        p.adjust.method = "none",
        detailed = TRUE
      )

    # 单独计算每个组的效应量以及95%置信区间
    cohens_d_results <- oxford_dist %>%
      group_by(oxford_c, classification) %>%
      group_modify(~{
        if (length(unique(.x$kmeans_cluster)) == 2) {
          # 计算Cohen's d及其95%置信区间
          effect_size <- effectsize::cohens_d(
            distance_value ~ kmeans_cluster,
            data = .x
          )

          # 提取Cohen's d和标准误
          cohens_d <- effect_size$Cohens_d

          # 计算95%置信区间
          ci_lower <- effect_size$CI_low
          ci_upper <- effect_size$CI_high

          # 返回效应量及置信区间
          return(data.frame(cohens_d = cohens_d, ci_lower = ci_lower, ci_upper = ci_upper))
        } else {
          # 如果kmeans_cluster的个数不是2，则返回NA
          return(data.frame(cohens_d = NA, ci_lower = NA, ci_upper = NA))
        }
      })


    # 合并效应量结果
    pairwise_results <- pairwise_results %>%
      left_join(cohens_d_results, by = c("oxford_c", "classification"))

    # 整体p值校正
    pairwise_results$p.adjoverall <- p.adjust(pairwise_results$p, method = "bonferroni")

    # 创建更丰富的显著性标记 - 符合nature要求
    pairwise_results <- pairwise_results %>%
      mutate(
        p.adj.signif = case_when(
          p.adjoverall < 0.001 ~ "***",
          p.adjoverall < 0.01 ~ "**",
          p.adjoverall < 0.05 ~ "*",
          TRUE ~ "NS"
        ),
        # nature通常要求明确的p值范围而非星号，但星号在图形中更简洁
        # 创建包含效应量的标签
        label_detailed = case_when(
          p.adjoverall < 0.001 ~ sprintf("P<0.001\n(d=%.2f)", cohens_d),
          p.adjoverall < 0.01 ~ sprintf("P<0.01\n(d=%.2f)", cohens_d),
          p.adjoverall < 0.05 ~ sprintf("P<0.05\n(d=%.2f)", cohens_d),
          TRUE ~ sprintf("P=%.3f\n(d=%.2f)", p.adjoverall, cohens_d)
        ),
        # 简化的标签版本 - 符合nature简洁要求
        label_simple = case_when(
          p.adjoverall < 0.001 ~ "***",
          p.adjoverall < 0.01 ~ "**",
          p.adjoverall < 0.05 ~ "*",
          TRUE ~ "NS"
        )
      )

    # 准备显著性标记数据 - 修正位置计算
    sig_data <- pairwise_results %>%
      left_join(
        oxford_dist %>%
          group_by(oxford_c, classification) %>%
          summarise(
            x_min = min(distance_value, na.rm = TRUE),
            x_max = max(distance_value, na.rm = TRUE),
            .groups = 'drop'
          ),
        by = c("oxford_c", "classification")
      ) %>%
      group_by(classification) %>%
      mutate(
        # 将oxford_c转换为数值用于y轴定位
        y_num = as.numeric(oxford_c) + 0.55,
        # 设置x位置为最大值右侧5%处
        x = x_min + 0.05 * (x_max - x_min)
      ) %>%
      mutate(
        p.adj.color = case_when(
          p.adjoverall < 0.05 ~ "Significant",
          TRUE ~ "Not Significant"
        )
      )

    # nature推荐的颜色方案 - 专业、清晰、色盲友好
    fill_colors <- c(
      "#2E86AB",  # 蓝色
      "#A23B72",  # 洋红色
      "#2E86AB",  # 蓝色
      "#A23B72"  # 洋红色
    )

    # 创建图形 - 符合nature风格
    # 精确控制版本
    oxford_dist$oxford_c <- factor(oxford_dist$oxford_c,
                                   levels = rev(levels(oxford_dist$oxford_c)))


    g_cluster <- ggplot(oxford_dist, aes(x = distance_value, y = oxford_c,
                                         fill = kmeans_cluster)) +

      # 1. 密度脊线图层 - 使用rel_heights参数控制
      geom_density_ridges(
        alpha = 0.7,
        scale = 0.9,
        color = "white",
        linewidth = 0.1,
        rel_min_height = 0.01  # 最小高度
      ) +

      # 2. 显著性点层
      geom_point(
        data = sig_data,
        aes(x = x, y = y_num, size = abs(cohens_d), color = p.adj.color),
        inherit.aes = FALSE,
        alpha = 0.8
      ) +

      # 3. 颜色比例尺
      scale_fill_manual(
        values = fill_colors,
        name = "Classification",
        labels = c("low-activity", "high-activity", "low-risk IgAN", "high-risk IgAN")
      ) +

      # 4. 点的大小比例尺
      scale_size_continuous(
        range = c(1, 6),
        name = "Effect Size \n(Cohen's d)",
        guide = guide_legend(override.aes = list(color = "grey20"))
      ) +

      # 5. 点的颜色比例尺
      scale_color_manual(
        values = c("Significant" = '#DE2910', "Not Significant" = "grey70"),
        name = "Significance",
        guide = guide_legend(override.aes = list(size = 3))
      ) +

      # 6. 标签
      labs(
        x = "Standardized Distance to Renal Pathology Embedding Vector",
        y = "Oxford Classification Patterns"
      ) +

      # 7. 分面设置
      facet_wrap(
        ~classification,
        ncol = 2,
        scales = "free_x"
      ) +

      # 8. 坐标轴设置
      scale_x_continuous(
        expand = expansion(mult = c(0.05, 0.15))
      ) +

      # 使用scale_y_discrete并控制expand参数
      scale_y_discrete(
        expand = expansion(mult = c(0.02, 0.04))  # 顶部15%扩展，底部2%扩展
      ) +

      # 9. 主题设置
      theme_ridges(font_size = 10, center_axis_labels = TRUE) +
      theme(
        # 文本设置
        text = element_text(family = "Arial", color = "black"),

        axis.line = element_line(color = "black", linewidth = 0.3),


        # 坐标轴标题
        axis.title = element_text(face = "bold", size = 9, color = "black"),
        axis.title.y = element_text(margin = margin(r = 8)),

        # 坐标轴文本
        axis.text = element_text(color = "black", size = 8),
        axis.text.y = element_text(
          margin = margin(r = 3),
          face = "bold",
          color = "black"
        ),

        # 图例设置
        legend.position = "right",
        legend.title = element_text(face = "bold", size = 9),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.4, "cm"),

        # 简洁strip设计 - 高度适中的版本
        strip.background = element_rect(
          fill = "grey95",  # 浅灰色背景
          color = NA,       # 无边框
          size = 0
        ),
        strip.text = element_text(
          face = "bold",
          size = 9.5,      # 稍微减小字号
          color = "black",
          margin = margin(
            t = 4,    # 减小上边距
            b = 4,    # 减小下边距
            l = 6,
            r = 6
          )
        ),

        # 调整面板间距
        panel.spacing = unit(0.7, "lines"),

        # 网格线
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.2),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey95", linewidth = 0.1),
        panel.grid.minor.y = element_blank(),

        # 整体边距 - 顶部增加更多空间
        plot.margin = margin(3, 1, 2, 1)
      )

    # 如果需要进一步调整y轴标签位置
    g_cluster <- g_cluster +
      theme(
        axis.text.y = element_text(
          vjust = 0.5,  # 垂直居中
          hjust = 1
        )
      )


    # 输出图形
    # print(g_cluster)

    ggsave(
      filename = "Results/oxford_feature_distance_distribution_full_qwen3_plus.pdf",
      plot = g_cluster,
      device = cairo_pdf,
      width = 6.5,
      height = 9.6
    )


  }
  ## chartGPT plot
  if (TRUE) {
    # 合并 sig_data 和 oxford_dist
    merged_data <- merge(oxford_dist, sig_data, by = c("oxford_c", "classification"))

    # 过滤缺失的效应大小
    merged_data <- merged_data[!is.na(merged_data$cohens_d),]

    save(merged_data, df,
         file = "dataPlots/oxford_distance_distribution_merged_qwen3_plus.Rdata")


    # Set nature-compliant theme
    theme_nature <- function(base_size = 11) {
      theme_minimal(base_size = base_size) %+replace%
        theme(
          text = element_text(family = "sans", color = "black"),
          axis.title = element_text(face = "bold", size = rel(1.1)),
          axis.text = element_text(color = "black"),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.ticks = element_line(color = "black", linewidth = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
          plot.caption = element_text(size = rel(0.8), color = "gray40", hjust = 0),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          strip.text = element_text(face = "bold")
        )
    }


    # 确保在ggplot中使用正确的分组变量
    p_violin <- ggplot(merged_data,
                       aes(y = reorder(oxford_c, cohens_d), # 确保按效应大小排序
                           x = distance_value,
                           fill = kmeans_cluster)) +

      # 使用violin图表示分布
      geom_violin(alpha = 0.75, color = "black", linewidth = 0.3, trim = TRUE,
                  position = position_dodge(width = 0.75)) +

      # 添加箱线图层，显示中位数和IQR
      geom_boxplot(width = 0.15, color = "white", outlier.shape = NA, linewidth = 0.2,
                   position = position_dodge(width = 0.75)) +

      # 添加散点图（可选）
      # geom_jitter(width = 0.1, size = 0.01, alpha = 0.2, color = "black") +

      # 主题和标签设置
      labs(
        x = "Embedding Distance",
        y = "Oxford Classification Patterns",
        fill = 'Classification'
      ) +

      scale_fill_manual(values = c("low-risk" = "lightblue", "high-risk" = "lightcoral",
                                   "low-activity" = "lightgreen", "high-activity" = "lightgoldenrodyellow")) +

      theme_minimal(base_size = 10) +
      theme(
        panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
        panel.grid.minor = element_blank(),
        text = element_text(family = "Arial", color = "black"),
        axis.title = element_text(face = "bold"),
        panel.grid = element_blank(),
        axis.line = element_line(color = "black", linewidth = 0.3),
        legend.position = "top",
        plot.margin = margin(1, 0, 3, 1)
      ) +
      # 面板分面设置
      facet_grid(
        . ~ classification,
        scales = "free_x",
        space = "free_x"
      )


    # Forest plot: 每个MEST-C pattern的效应大小与CI
    p_forest <- ggplot(merged_data,
                       aes(y = reorder(oxford_c, cohens_d), x = cohens_d)) +

      # 绘制效应大小（Cohen's d）和95% CI
      geom_pointrange(
        aes(xmin = ci_lower, xmax = ci_upper),
        size = 0.1,
        fatten = 1.5
      ) +

      # 添加垂直的零线
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.3) +

      # 设置标签
      labs(
        x = "Effect Size (Cohen's d)",
        y = NULL
      ) +

      # 面板分面设置，统一x轴范围
      facet_grid(
        . ~ classification,
        scales = "fixed",  # 保持 x 轴范围一致
        space = "free_x"
      ) +

      # 设置x轴范围，确保一致性
      scale_x_continuous(
        limits = c(min(merged_data$ci_lower, na.rm = TRUE) - 0.1, max(merged_data$ci_upper, na.rm = TRUE) + 0.1),  # 设置统一范围
        expand = c(0.05, 0.05)  # 在两端留一些空间
      ) +

      # 主题和标签
      theme_minimal(base_size = 10) +
      theme(
        panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
        panel.grid.minor = element_blank(),
        text = element_text(family = "Arial", color = "black"),
        axis.title = element_text(face = "bold"),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        panel.grid = element_blank(),
        axis.line = element_line(color = "black", linewidth = 0.3),
        plot.margin = margin(1, 1, 3, 0)
      )


    p_main <- cowplot::plot_grid(
      p_violin, p_forest,
      ncol = 2,
      align = "h",
      labels = c("A", "B"),
      rel_widths = c(1.3, 1)
    )

    Cairo::CairoPDF("results/Figure4_Morphology_Risk_Orientation_nature.pdf",
                    width = 7.5,
                    height = 12)
    print(p_main)
    dev.off()


    if (FALSE) {
      g_cluster <- ggplot(merged_data, aes(x = distance_value, y = oxford_c,
                                           fill = kmeans_cluster)) +
        # 1. 森林图图层 - 展示效应值和置信区间（放在最底层）
        geom_segment(
          aes(x = ci_lower, xend = ci_upper,
              y = oxford_c, yend = oxford_c),
          color = "grey40",
          linewidth = 0.4,
          alpha = 0.6,
          position = position_nudge(y = -0.15),  # 向下偏移，放在密度图下方
          inherit.aes = FALSE
        ) +
        geom_point(
          aes(x = cohens_d, y = oxford_c,
              size = abs(cohens_d),
              color = ifelse(p.adj < 0.05, "Significant", "Not Significant")),
          shape = 16,
          position = position_nudge(y = -0.15),  # 向下偏移，与线段对齐
          inherit.aes = FALSE
        ) +
        geom_vline(
          xintercept = 0,
          linetype = "dashed",
          color = "grey50",
          linewidth = 0.25,
          alpha = 0.7
        ) +
        # 2. 密度脊线图层
        geom_density_ridges(
          alpha = 0.7,
          scale = 0.9,
          color = "white",
          linewidth = 0.1,
          rel_min_height = 0.01
        ) +
        # 3. 颜色比例尺 - 分类填充色
        scale_fill_manual(
          values = fill_colors,
          name = "Classification",
          labels = c("low-activity", "high-activity", "low-risk IgAN", "high-risk IgAN")
        ) +
        # 4. 点的大小比例尺 - 效应值大小
        scale_size_continuous(
          range = c(1, 4),
          name = "Effect Size \n(Cohen's d)",
          guide = guide_legend(override.aes = list(color = "grey20", shape = 16))
        ) +
        # 5. 点的颜色比例尺 - 显著性
        scale_color_manual(
          values = c("Significant" = '#DE2910', "Not Significant" = "grey70"),
          name = "Significance",
          guide = guide_legend(override.aes = list(size = 2.5, shape = 16))
        ) +
        # 6. 标签
        labs(
          x = "Standardized Distance to Renal Pathology Embedding Vector",
          y = "Oxford Classification Patterns"
        ) +
        # 7. 分面设置
        facet_wrap(
          ~classification,
          ncol = 2,
          scales = "free_x"
        ) +
        # 8. 坐标轴设置
        scale_x_continuous(
          expand = expansion(mult = c(0.05, 0.15))
        ) +
        # 9. 调整y轴比例 - 为森林图留出空间
        scale_y_discrete(
          expand = expansion(mult = c(0.08, 0.04))  # 增加底部扩展，为森林图留空间
        ) +
        # 10. 主题设置
        theme_ridges(font_size = 10, center_axis_labels = TRUE) +
        theme(
          # 文本设置
          text = element_text(family = "Arial", color = "black"),
          axis.line = element_line(color = "black", linewidth = 0.3),

          # 坐标轴标题
          axis.title = element_text(face = "bold", size = 9, color = "black"),
          axis.title.y = element_text(margin = margin(r = 8)),

          # 坐标轴文本
          axis.text = element_text(color = "black", size = 8),
          axis.text.y = element_text(
            margin = margin(r = 3),
            face = "bold",
            color = "black"
          ),

          # 图例设置
          legend.position = "right",
          legend.title = element_text(face = "bold", size = 9),
          legend.text = element_text(size = 8),
          legend.key.size = unit(0.4, "cm"),
          legend.box.spacing = unit(0.3, "cm"),

          # 分面标签设计
          strip.background = element_rect(
            fill = "grey95",
            color = NA,
            size = 0
          ),
          strip.text = element_text(
            face = "bold",
            size = 9.5,
            color = "black",
            margin = margin(t = 4, b = 4, l = 6, r = 6)
          ),

          # 调整面板间距
          panel.spacing = unit(0.7, "lines"),

          # 网格线
          panel.grid.major.x = element_line(color = "grey90", linewidth = 0.2),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),  # 移除y轴主要网格线
          panel.grid.minor.y = element_blank(),

          # 整体边距
          plot.margin = margin(3, 1, 2, 1)
        )

      # 如果需要更精细的控制，可以添加注释说明森林图的位置
      g_cluster <- g_cluster +
        annotate("text",
                 x = min(oxford_dist$distance_value, na.rm = TRUE),
                 y = -0.5,
                 label = "Effect Size (Cohen's d) with 95% CI",
                 hjust = 0,
                 vjust = 0,
                 size = 2.5,
                 color = "grey40",
                 family = "Arial")
    }

    # 输出图形
    ggsave(
      filename = "results/Figure4_Morphology_Risk_Orientation_nature_Combined.pdf",
      plot = g_cluster,
      device = cairo_pdf,
      width = 7.5,
      height = 9.6
    )


  }
}
## 评价一致性分析，包括外部数据
if (TRUE) {
  ## data prepare
  if (TRUE) {
    load("data/mestc_distance_difference_sign_first_hospital.Rdata")
    load("data/mestc_true_and_distance_difference_sign_second_hospital.Rdata")
    mestc_dist <- mestc_dist_first_hospital %>%
      mutate(group = "true_mestc") %>%
      select(-diff_sign) %>%
      bind_rows(
        pairwise_results %>%
          rename(mestc = oxford_c,
                 pct_diff = cohens_d) %>%
          mutate(
            classification = case_when(
              classification == "Pathology Subtype" ~ "Pathology Subtypes",
              classification == "Clinical Risk Stratification" ~ "Clinical Risk Stratification",
              TRUE ~ NA_character_
            )
          ) %>%
          mutate(
            pct_diff = ifelse(pct_diff > 0, "positive", "negative"),
            group = "distance_difference") %>%
          select(mestc, pct_diff, group, classification) %>%
          distinct()
      ) %>%
      mutate(
        hospital = "first"
      ) %>%
      bind_rows(
        mestc_dist_second_hospital
      ) %>%
      mutate(
        mestc = as.factor(as.character(mestc))
      )


    # Create complete rater identifiers
    mestc_dist <- mestc_dist %>%
      mutate(
        rater_id = paste(hospital, group, classification, sep = "_"),
        rater_label = paste(
          substr(hospital, 1, 1),  # f or s
          substr(group, 1, 3),      # dis or tru
          substr(gsub(" ", "", classification), 1, 4),  # Clin or Path
          sep = "_"
        )
      )

    # Convert to wide format
    wide_data <- mestc_dist %>%
      select(mestc, rater_label, pct_diff) %>%
      pivot_wider(
        names_from = rater_label,
        values_from = pct_diff
      )

    # Define all comparison combinations
    comparisons <- list(
      # 1. Same hospital, same condition, different methods
      list(name = "SameHosp_SameCond_DiffMethod_dis_first",
           raters = c("f_dis_Clin", "f_dis_Path"),
           description = "First hospital, distance condition, different 危险分层方法"),
      list(name = "SameHosp_SameCond_DiffMethod_tru_first",
           raters = c("f_tru_Clin", "f_tru_Path"),
           description = "First hospital, true condition, different 危险分层方法"),
      list(name = "SameHosp_SameCond_DiffMethod_dis_second",
           raters = c("s_dis_Clin", "s_dis_Path"),
           description = "Second hospital, distance condition, different 危险分层方法"),
      list(name = "SameHosp_SameCond_DiffMethod_tru_second",
           raters = c("s_tru_Clin", "s_tru_Path"),
           description = "Second hospital, true condition, different 危险分层方法"),

      # 2. Same hospital, same method, different conditions
      list(name = "SameHosp_SameMethod_DiffCond_Clin_first",
           raters = c("f_dis_Clin", "f_tru_Clin"),
           description = "First hospital, Clinical method, different mestc 相关方法"),
      list(name = "SameHosp_SameMethod_DiffCond_Path_first",
           raters = c("f_dis_Path", "f_tru_Path"),
           description = "First hospital, Pathology method, different mestc 相关方法"),
      list(name = "SameHosp_SameMethod_DiffCond_Clin_second",
           raters = c("s_dis_Clin", "s_tru_Clin"),
           description = "Second hospital, Clinical method, different mestc 相关方法"),
      list(name = "SameHosp_SameMethod_DiffCond_Path_second",
           raters = c("s_dis_Path", "s_tru_Path"),
           description = "Second hospital, Pathology method, different mestc 相关方法"),

      # 3. Same condition, same method, different hospitals
      list(name = "SameCond_SameMethod_DiffHosp_dis_Clin",
           raters = c("f_dis_Clin", "s_dis_Clin"),
           description = "Distance condition, Clinical method, different hospitals"),
      list(name = "SameCond_SameMethod_DiffHosp_dis_Path",
           raters = c("f_dis_Path", "s_dis_Path"),
           description = "Distance condition, Pathology method, different hospitals"),
      list(name = "SameCond_SameMethod_DiffHosp_tru_Clin",
           raters = c("f_tru_Clin", "s_tru_Clin"),
           description = "True condition, Clinical method, different hospitals"),
      list(name = "SameCond_SameMethod_DiffHosp_tru_Path",
           raters = c("f_tru_Path", "s_tru_Path"),
           description = "True condition, Pathology method, different hospitals")
    )


    # 改进后的代码 - 修正标签提取逻辑
    results_df <- lapply(comparisons, function(comp) {
      # Extract data
      comp_data <- wide_data[, comp$raters]
      comp_data_complete <- comp_data[complete.cases(comp_data),]

      # Calculate Fleiss' Kappa
      kappa_result <- kappam.fleiss(as.matrix(comp_data_complete))

      # Extract results
      kappa_val <- kappa_result$value
      p_val <- kappa_result$p.value
      z_val <- kappa_result$statistic

      # 改进：使用更准确的逻辑判断比较类型
      comp_type <- if (grepl("DiffMethod", comp$name)) {
        "Different Methods"
      } else if (grepl("DiffCond", comp$name)) {
        "Different Conditions"
      } else if (grepl("DiffHosp", comp$name)) {
        "Different Hospitals"
      } else {
        "Unknown"
      }

      # 改进：直接从description中提取信息
      desc <- comp$description

      # 提取医院信息
      hospital_info <- if (grepl("First hospital", desc)) {
        "First"
      } else if (grepl("Second hospital", desc)) {
        "Second"
      } else if (grepl("Distance condition|True condition", desc)) {
        # 对于跨医院比较，从raters中提取医院信息
        raters <- comp$raters
        if (all(grepl("^f_", raters))) {
          "First"
        } else if (all(grepl("^s_", raters))) {
          "Second"
        } else {
          "Cross-hospital"
        }
      } else {
        "Unknown"
      }

      # 提取条件信息
      condition_info <- if (grepl("distance condition", desc)) {
        "Distance"
      } else if (grepl("true condition", desc)) {
        "True"
      } else if (grepl("Distance condition", desc)) {
        "Distance"
      } else if (grepl("True condition", desc)) {
        "True"
      } else {
        # 从raters名称中提取
        if (any(grepl("_dis_", comp$raters))) "Distance"
        else if (any(grepl("_tru_", comp$raters))) "True"
        else "Mixed"
      }

      # 提取方法信息
      method_info <- if (grepl("Clinical method", desc)) {
        "Clinical"
      } else if (grepl("Pathology method", desc)) {
        "Pathology"
      } else if (grepl("Clinical", desc)) {
        "Clinical"
      } else if (grepl("Pathology", desc)) {
        "Pathology"
      } else {
        # 从raters名称中提取
        if (any(grepl("_Clin", comp$raters))) "Clinical"
        else if (any(grepl("_Path", comp$raters))) "Pathology"
        else "Mixed"
      }

      # Agreement level classification (Landis & Koch)
      agreement_level <- if (is.na(kappa_val)) {
        "Cannot calculate"
      } else if (kappa_val < 0) {
        "Worse than chance"
      } else if (kappa_val < 0.2) {
        "Slight"
      } else if (kappa_val < 0.4) {
        "Fair"
      } else if (kappa_val < 0.6) {
        "Moderate"
      } else if (kappa_val < 0.8) {
        "Substantial"
      } else {
        "Almost perfect"
      }

      # Significance markers
      sig_mark <- if (is.na(p_val)) {
        "NA"
      } else if (p_val < 0.001) {
        "***"
      } else if (p_val < 0.01) {
        "**"
      } else if (p_val < 0.05) {
        "*"
      } else {
        "ns"
      }

      # 改进：根据比较类型生成有意义的DisplayLabel
      display_label <- switch(comp_type,
                              "Different Methods" = paste("Different Methods:", condition_info,
                                                          " (", hospital_info, ")", sep = ""),
                              "Different Conditions" = paste("Different Conditions:", method_info,
                                                             " (", hospital_info, ")", sep = ""),
                              "Different Hospitals" = paste("Different Hospitals:", condition_info,
                                                            " (", method_info, ")", sep = ""),
                              paste(comp_type, ":", condition_info, " (", method_info, ")", sep = "")
      )

      # Return results
      data.frame(
        ComparisonID = comp$name,
        Description = comp$description,
        ComparisonType = comp_type,
        Hospital = hospital_info,
        Condition = condition_info,
        Method = method_info,
        N = nrow(comp_data_complete),
        Kappa = round(kappa_val, 3),
        Z = round(z_val, 2),
        PValue = p_val,
        Significance = sig_mark,
        AgreementLevel = agreement_level,
        DisplayLabel = display_label,
        stringsAsFactors = FALSE
      )
    })
    # 将所有结果合并为一个数据框
    results_df <- do.call(rbind, results_df)
    # Sort results by Kappa value
    results_df <- results_df[order(-results_df$Kappa),]

    new_labels <- c(
      # Inter-hospital
      "Different Hospitals:True (Pathology)" = "MEST-C (Pathology)",
      "Different Hospitals:True (Clinical)" = "MEST-C (Clinical)",
      "Different Hospitals:Distance (Pathology)" = "Similarity (Pathology)",
      "Different Hospitals:Distance (Clinical)" = "Similarity (Clinical)",

      # Inter-method
      "Different Methods:True (First)" = "MEST-C (First)",
      "Different Methods:True (Second)" = "MEST-C (Second)",
      "Different Methods:Distance (First)" = "Similarity (First)",
      "Different Methods:Distance (Second)" = "Similarity (Second)",

      # Inter-condition
      "Different Conditions:Clinical (First)" = "Clinical (First)",
      "Different Conditions:Clinical (Second)" = "Clinical (Second)",
      "Different Conditions:Pathology (First)" = "Pathology (First)",
      "Different Conditions:Pathology (Second)" = "Pathology (Second)"
    )


    ComparisonDomain <- c(
      "Different Hospitals" = "Inter-hospital",
      "Different Methods" = "Inter-method",
      "Different Conditions" = "Inter-condition"
    )

    results_df$DisplayLabel <- new_labels[results_df$DisplayLabel]
    results_df$ComparisonType <- ComparisonDomain[results_df$ComparisonType]

    print(results_df)
  }
  ### analysis
  if (TRUE) {


    # Set nature-compliant theme
    theme_nature <- function(base_size = 11) {
      theme_minimal(base_size = base_size) %+replace%
        theme(
          text = element_text(family = "sans", color = "black"),
          axis.title = element_text(face = "bold", size = rel(1.1)),
          axis.text = element_text(color = "black"),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.ticks = element_line(color = "black", linewidth = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
          plot.caption = element_text(size = rel(0.8), color = "gray40", hjust = 0),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          strip.text = element_text(face = "bold"),
          plot.margin = margin(15, 15, 15, 15)
        )
    }

    # Figure 1: Bar plot of Kappa values by comparison type
    p1 <- ggplot(results_df, aes(x = reorder(DisplayLabel, Kappa), y = Kappa,
                                 fill = ComparisonType)) +
      geom_bar(stat = "identity", width = 0.7, color = "black", linewidth = 0.3) +
      geom_hline(yintercept = c(0, 0.2, 0.4, 0.6, 0.8),
                 linetype = c("solid", "dotted", "dotted", "dotted", "dotted"),
                 color = "gray50", linewidth = 0.3) +
      geom_text(aes(label = ifelse(PValue < 0.05,
                                   paste0(round(Kappa, 2), Significance),
                                   round(Kappa, 2))),
                hjust = ifelse(results_df$Kappa > 0, -0.2, 1.2),
                size = 3, color = "black") +
      scale_fill_manual(values = c("Inter-method" = "#D9D9D9",
                                   "Inter-condition" = "#969696",
                                   "Inter-hospital" = "#525252")) +
      scale_y_continuous(limits = c(-0.5, 1.1), breaks = seq(-0.4, 1.00, 0.2)) +
      coord_flip() +
      labs(
        x = NULL,
        y = "Fleiss’ Kappa Coefficient",
        fill = "Validation Domain") +
      theme_nature() +
      theme(axis.text.y = element_text(size = 9),
            legend.position = "right")

    # print(p1)

    # # Save as high-resolution TIFF for publication
    ggsave("results/Figure1_Kappa_Comparisons.pdf", p1,
           width = 8, height = 6, dpi = 300)


  }


}
