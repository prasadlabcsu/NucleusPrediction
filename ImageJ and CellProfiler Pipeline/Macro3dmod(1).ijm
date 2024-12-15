setBatchMode(true);

// Define the input and output directories
inputDir = "/home/spag/Downloads/3dCTL/";
outputDir = "/home/spag/Downloads/CTLoutputFromImageJ/";

// Get a list of all .czi files in the input directory
fileList = getFileList(inputDir);

// Iterate through each file in the directory
for (i = 0; i < fileList.length; i++) {
    if (endsWith(fileList[i], ".czi")) {
        // Set the current file name
        fileName = fileList[i];
        fullPath = inputDir + fileName;
        

        // Process the current file
        processFile(fullPath, fileName, outputDir);
    }
}

function printOpenImages() {
    print("--- Open Images ---");
    imageList = getList("image.titles");
    if (imageList.length == 0) {
        print("No images are currently open.");
    } else {
        for (i = 0; i < imageList.length; i++) {
            print((i+1) + ": " + imageList[i]);
            
        }
    }
    print("-------------------");
}

// Function to process a single file
function processFile(fullPath, fileName, outputDir) {
    run("Bio-Formats Importer", "open=[" + fullPath + "] autoscale color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
    
    printOpenImages();
    selectImage(fileName+" - C=2");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");  
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Enhance Contrast", "saturated=0.35");
    
    //setAutoThreshold("Default dark no-reset");
    run("Threshold...");
    setThreshold(40,255);
    setOption("BlackBackground", true);
    run("Convert to Mask", "background=Dark black");
    
    run("Gaussian Blur...", "sigma=15 stack");
    run("Threshold...");
    setThreshold(8,255);
    run("Convert to Mask", "background=Dark black");
    selectImage(fileName+" - C=1");

    selectImage(fileName+" - C=0");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");  
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Next Slice [>]");
    run("Enhance Contrast", "saturated=0.35");
    
    setThreshold(25,255);
    run("Convert to Mask", "background=Dark black");
    run("Gaussian Blur...", "sigma=19 stack");
    //setAutoThreshold("Default dark no-reset");
    run("Threshold...");
    setThreshold(12,255);
    run("Convert to Mask", "background=Dark black");
    printOpenImages();
    imageCalculator("Add create stack", fileName+" - C=0", fileName+" - C=2");
    selectImage("Result of " + fileName+" - C=0");
    run("Gaussian Blur...", "sigma=20 stack");
    //setAutoThreshold("Default dark no-reset");
    run("Threshold...");
    setThreshold(15, 255);
    run("Convert to Mask", "background=Dark black");

    run("Gaussian Blur...", "sigma=20 stack");
    //setAutoThreshold("Default dark no-reset");
    run("Threshold...");
    setThreshold(30,255);
    run("Convert to Mask", "background=Dark black");

    selectImage(fileName+" - C=0");
    close();
    selectImage(fileName+" - C=2");
    close();
    // Function to split a stack in half and return the bottom half
function splitStackTopHalf(title, newTitle) {
    selectImage(title);
    slices = nSlices;
    mid = floor(slices / 3);
    run("Make Substack...", "slices=" + (mid+1) + "-" + slices);
    rename(newTitle);
    return newTitle;
}

// Function to split a stack in half and return the top half
function splitStackBottomHalf(title, newTitle) {
    selectImage(title);
    slices = nSlices;
    mid = floor(slices / 3);
    run("Make Substack...", "slices=1-" + mid);
    rename(newTitle);
    return newTitle;
}

// Main macro
maskTitle = "Result of " + fileName + " - C=0";
selectImage(maskTitle);
run("Fill Holes", "stack");

// Split the mask (we only need the bottom half)
maskBottomHalfTitle = splitStackBottomHalf(maskTitle, "MaskBottomHalf");

// Open and split the original image
run("Bio-Formats Importer", "open=[" + fullPath + "] autoscale color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
printOpenImages();

selectImage(fileName + " - C=1");
close();

// Process C0 channel
origTitle = fileName + " - C=0";
selectImage(origTitle);
rename("C1--" + fileName);
origTitle = "C1--" + fileName;
origTopHalfTitle = splitStackTopHalf(origTitle, "OrigTopHalf");
origBottomHalfTitle = splitStackBottomHalf(origTitle, "OrigBottomHalf");

// Process only the bottom half
imageCalculator("AND create stack", origBottomHalfTitle, maskBottomHalfTitle);
processedBottomHalfTitle = "ProcessedBottomHalf";
rename(processedBottomHalfTitle);

// Recombine with unprocessed top half
run("Concatenate...", "image1=" + processedBottomHalfTitle + " image2=" + origTopHalfTitle + " title=ProcessedC1");

// Process C2 (nuclei channel) similarly
nucTitle = fileName + " - C=2";
nucTopHalfTitle = splitStackTopHalf(nucTitle, "NucTopHalf");
nucBottomHalfTitle = splitStackBottomHalf(nucTitle, "NucBottomHalf");

imageCalculator("AND create stack", nucBottomHalfTitle, maskBottomHalfTitle);
processedNucBottomHalfTitle = "ProcessedNucBottomHalf";
rename(processedNucBottomHalfTitle);

run("Concatenate...", "image1=" + processedNucBottomHalfTitle + " image2=" + nucTopHalfTitle + " title=ProcessedNuc");

// Save results
selectImage("ProcessedC1");
saveAs("Tiff", outputDir + File.separator + "  " + i);

selectImage("ProcessedNuc");
saveAs("Tiff", outputDir + File.separator + "  nuc " + i);

// Clean up
run("Close All");
}
