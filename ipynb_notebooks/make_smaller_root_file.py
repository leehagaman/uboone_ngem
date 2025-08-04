import subprocess
import os

# 1) The ROOT macro code as a Python multiline string
root_macro = r'''
int ProcessDirectory(TDirectory *input_dir, TDirectory *output_dir, const char* dir_path = "") {
    int trees_copied = 0;
    
    TIter nextkey(input_dir->GetListOfKeys());
    TKey *key;
    
    while ((key = (TKey*)nextkey())) {
        const char *keyname = key->GetName();
        TString full_path = TString(dir_path) + "/" + keyname;
        
        printf("Processing: %s\n", full_path.Data());
        
        TObject *obj = key->ReadObj();
        
        // Check if it's a TTree
        TTree *tree = dynamic_cast<TTree*>(obj);
        if (tree) {
            Long64_t nentries = tree->GetEntries();
            printf("  Found TTree '%s' with %lld entries\n", keyname, nentries);
            
            Long64_t copy_entries = (nentries < 1000) ? nentries : 1000;
            printf("  Copying first %lld entries...\n", copy_entries);
            
            output_dir->cd();
            TTree *newtree = tree->CopyTree("", "", copy_entries);
            if (newtree) {
                newtree->Write();
                printf("  Wrote new tree '%s' with %lld entries\n", newtree->GetName(), newtree->GetEntries());
                trees_copied++;
            } else {
                printf("  ERROR: CopyTree returned nullptr\n");
            }
        }
        // Check if it's a directory
        else if (obj->InheritsFrom("TDirectory")) {
            TDirectory *subdir = dynamic_cast<TDirectory*>(obj);
            if (subdir) {
                printf("  Found directory '%s', recursing...\n", keyname);
                
                // Create corresponding directory in output file
                output_dir->cd();
                TDirectory *output_subdir = output_dir->mkdir(keyname);
                
                // Recursively process the subdirectory
                trees_copied += ProcessDirectory(subdir, output_subdir, full_path.Data());
            }
        }
        else {
            printf("  Skipping object '%s' of class %s\n", keyname, obj->ClassName());
        }
    }
    
    return trees_copied;
}

void copy_first_1000(const char* input_filename = "input.root",
                    const char* output_filename = "small_sample.root") {
    TFile *f_in = TFile::Open(input_filename, "READ");
    if (!f_in || f_in->IsZombie()) {
        printf("Error: Cannot open input file '%s'\n", input_filename);
        return;
    }
    printf("Opened input file: %s\n", input_filename);

    TFile *f_out = TFile::Open(output_filename, "RECREATE");
    if (!f_out || f_out->IsZombie()) {
        printf("Error: Cannot create output file '%s'\n", output_filename);
        f_in->Close();
        return;
    }
    printf("Created output file: %s\n", output_filename);

    int total_trees = ProcessDirectory(f_in, f_out, "");
    
    printf("Finished copying %d trees.\n", total_trees);

    f_out->Close();
    f_in->Close();
}
'''

# 2) Write it to a file
with open("copy_first_1000.C", "w") as f:
    f.write(root_macro)

# 3) Run the macro with ROOT from the command line (replace filenames here)
input_file = "../data_files/SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_beam_on_may8_reco2_hist_goodruns_62280934_snapshot.root"
output_file = "../data_files/smaller_SURPRISE_BNB_data_file.root"

# Use conda run to execute ROOT in the specified environment
cmd = [
    "conda", "run", "-n", "root-env-225",
    "root", "-b", "-q", 
    f'copy_first_1000.C("{input_file}","{output_file}")'
]

print("Running ROOT macro...")
result = subprocess.run(cmd, capture_output=True, text=True)

os.remove("copy_first_1000.C")

# 4) Print ROOT output
print(result.stdout)
print(result.stderr)
