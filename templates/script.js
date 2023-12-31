document.addEventListener("DOMContentLoaded", function() {
    var inputFile = document.getElementById("file");
    var inputVideo = document.getElementById("input-video");
    var processedVideo = document.getElementById("processed-video");
    var plotCanvas = document.getElementById("plot-canvas");

    // Add an event listener to the form's submit event
    document.getElementById("form").addEventListener("submit", function(event) {
        // Prevent the default form submission
        event.preventDefault();

        // Create a FormData object with the form data
        var formData = new FormData(event.target);

        // Send a POST request to the API with the form data
        fetch("/process_video", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Set the source of the input video
            inputVideo.src = data.input_video;

            // Set the source of the processed video
            processedVideo.src = data.processed_video;

            // Draw the plot on the canvas
            drawPlot(data.plot_data);

            // Display additional results
            document.getElementById("results").innerHTML = JSON.stringify(data);
        })
        .catch(error => console.log(error));
    });

    function drawPlot(plotData) {
        var ctx = plotCanvas.getContext("2d");
        var width = plotCanvas.width;
        var height = plotCanvas.height;

        // Clear the canvas
        ctx.clearRect(0, 0, width, height);

        // Draw the plot data
        ctx.beginPath();
        ctx.moveTo(0, height - plotData[0]);

        for (var i = 1; i < plotData.length; i++) {
            var x = (i / plotData.length) * width;
            var y = height - plotData[i];
            ctx.lineTo(x, y);
        }

        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
});