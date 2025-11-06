
library(tidyverse, quietly = TRUE)
library(igraph)
require(assertthat)

# Files
cat("Choose the csv file/n")
df_path = file.choose()
df_base_path = basename(df_path)

cat("Choose Graphml file/n")
ml_path = file.choose()


repeat {
    # Ask for input
    user_input <- tolower(readline(prompt = "Choose File Type of csv file: New [y] or old [n] "))
    
    # Check if the input is valid
    if (user_input %in% c("y", "n")) {
      file_structure = ifelse(user_input == "y", "adni", "old")   # to capture different columns if it's ADNI data vs tthompson data
      break
    } else {
      cat("Invalid input. Please enter 'y' for Yes or 'N' for No.\n")
    }
  }


# DF 45 or 1451
df = read_csv(df_path, 
				col_types = cols(
					RID = col_integer()
				))

original_df_names = names(df)
names_df = names(df)  %>% str_to_lower()  %>%  # names that must be mapped
	str_replace(., "right\\_", "rh_")  %>% str_replace(.,"left\\_", "lh_")
names(df) <- names_df


# graphml file
graph = igraph::read_graph(ml_path, format = "graphml")
node_df = tibble(as.data.frame(vertex.attributes(graph)))
edge_df = graph %>% 
	edge.attributes() %>%
	as.data.frame() %>%
	cbind(as.data.frame(as_edgelist(graph)), .) %>%
	tibble() # results in first 2 variables being V1 and V2
names(edge_df) = c("source", "target", names(edge_df)[3:length(names(edge_df))])


# clean up graphml file names so that it canbe mapped with ADNI name with minimal regex
fs_names_processing <- function(list_to_process){
 	list_to_process %>%
       str_to_lower()  %>% 
		str_replace(., "lh\\.|left-", "lh_")  %>% 
		str_replace(., "rh\\.|right-", "rh_") %>% 
		str_replace(., "brain-stem", "brainstem") %>% 
		str_replace(.,"-", "_") %>% 
       return()
}


### Mapping procedure

fsnames <- node_df$dn_name %>% c() %>% fs_names_processing() # converted_names_to_be_mapped_into
dnnames <- node_df$dn_name

# Map the names together
map_fsnames = sapply(fsnames, function(x) grepv(x, names_df), simplify = F)
fsnames_unlisted = map_fsnames %>% unlist() %>% unname()

# sanity check
map_fsnames_ln = sapply(fsnames, function(x) grepv(x, names_df) %>% length(), simplify = F)
if (!see_if(are_equal(
	{map_fsnames_ln %>% unlist() %>% unname() == 2} %>% sum(), 
	83))) {
	warning("Sanity Check Failed More/Less items than necessary: Items Did not match")
}

# Mapping saved in a separate data frame
map_result_df <- lapply(names(map_fsnames), function(fsname) {
  suvr_name <- grepv("suvr", map_fsnames[[fsname]])[1] 
  volume_name <-grepv("volume", map_fsnames[[fsname]])[1] 
	
  data.frame(
    fsname = fsname,
    suvr_name = suvr_name,
    volume_name = volume_name,
    stringsAsFactors = FALSE
  )
})

map_result_df %<>% 
  bind_rows() %>% 
  tibble() %>% 
  bind_cols(dnnames = node_df$dn_name)


### Save the files to New folder
new_folder_name = new_folder_name = paste0("structured_files_", 
												df_base_path %>% basename() %>% tools::file_path_sans_ext())

structure_file_path = df_path %>% 
	dirname() %>% 
	file.path(., new_folder_name, fsep = .Platform$file.sep)
dir.create(structure_file_path, recursive = F, showWarnings = T)


# Capture column structure difference between new/old file types
if (file_structure == "adni") {
  cols_to_keep <- str_to_lower(c("LONIUID", "PTID", "RID", "SCANDATE", "QC_FLAG", "TRACER", "AMYLOID_STATUS", 
  								 "AMYLOID_STATUS_COMPOSITE_REF", "CENTILOIDS", "SUMMARY_SUVR", "SUMMARY_VOLUME", 
								 "WHOLECEREBELLUM_SUVR", "WHOLECEREBELLUM_VOLUME", "COMPOSITE_REF_SUVR", "COMPOSITE_REF_VOLUME", 
								 "CEREBELLUM_CORTEX_SUVR",	"CEREBELLUM_CORTEX_VOLUME",	"ERODED_SUBCORTICALWM_SUVR",	"ERODED_SUBCORTICALWM_VOLUME")
								)
} else {
  cols_to_keep <- str_to_lower(c("RID"))
}

### 

# SUVR VOLUME and Other Columns
for (var in c("suvr", "volume", "other")) {

	if (var == "other") {
		df %>% 
			select(-all_of(fsnames_unlisted)) %>% 
			write_csv(., 
				file = file.path(structure_file_path, 
									paste0(tools::file_path_sans_ext(df_base_path), "_", var,  ".csv"), 
									fsep = .Platform$file.sep)
				)
	} else{
		selected_col_names = fsnames_unlisted %>% grepv(paste0("_", var), .)
      
		map_list = map_result_df[[paste0(var, "_name")]] %>% c()
		map_list = append(map_list, cols_to_keep)
		names(map_list) = c(map_result_df$dnnames, cols_to_keep)

		df %>% 
			select(all_of(c(cols_to_keep, selected_col_names))) %>% 
          	rename_with(~ sapply(.x, function(y) names(which(map_list == y)))) %>% 
			write_csv(., 
					file = file.path(structure_file_path, 
									paste0(tools::file_path_sans_ext(df_base_path), "_", var,  ".csv"), 
									fsep = .Platform$file.sep)
			)
	}
}

map_result_df %>% 
	write_csv(., file = file.path(structure_file_path, 
									paste0(tools::file_path_sans_ext(df_base_path), "_", "mapping",  ".csv"), 
									fsep = .Platform$file.sep)
	)



# Sanity check: Check if suvr_name contains <prefix>_<fsname>_suvr and volume_name contains <prefix><fsname>_volume
suvr_check <- mapply(function(fsname, suvr_name) {
  grepl(paste0(".*", fsname, "_suvr$"), suvr_name)
}, map_result_df$fsname, map_result_df$suvr_name)

volume_check <- mapply(function(fsname, volume_name) {
  grepl(paste0(".*", fsname, "_volume$"), volume_name)
}, map_result_df$fsname, map_result_df$volume_name)


if (!see_if(are_equal(sum(suvr_check), 83))) {
	warning("Sanity Check of name order Failed: suvr")
}

if (!see_if(are_equal(sum(volume_check), 83))) {
	warning("Sanity Check of name order Failed: volume")
}