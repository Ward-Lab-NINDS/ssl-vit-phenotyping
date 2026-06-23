// Open SSL patch-test TIFFs in Fiji/ImageJ.
// Usage:
// 1. Open Fiji.
// 2. File > Open... this macro.
// 3. Click Run.
// 4. Choose an output folder such as outputs/ssl_patch_test_200.

outputDir = getDirectory("Choose SSL patch-test output folder");
patchDir = outputDir + "patches/";
manifestPath = outputDir + "patch_manifest.tsv";
phenotypePath = outputDir + "patch_phenotypes.tsv";
maxImages = 24;

print("SSL patch test output folder: " + outputDir);
print("Patch folder: " + patchDir);
print("Patch manifest: " + manifestPath);
print("Patch phenotype table: " + phenotypePath);

list = getFileList(patchDir);
opened = 0;
setBatchMode(true);
for (i = 0; i < list.length && opened < maxImages; i++) {
    if (!endsWith(list[i], ".tif") && !endsWith(list[i], ".tiff")) {
        continue;
    }
    open(patchDir + list[i]);
    run("Enhance Contrast", "saturated=0.35 normalize");
    run("Grays");
    rename(list[i]);
    opened++;
}
setBatchMode(false);

if (opened > 0) {
    run("Tile");
} else {
    showMessage("No patch TIFFs found", "No .tif or .tiff files were found in:\n" + patchDir);
}
