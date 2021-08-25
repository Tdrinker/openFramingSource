
/* * * * * */
/*  DATA   */
/* * * * * */


// moved here for readability
function getPreprocessingOptions() {
   return {
       remove_stopwords: $('#dc-stops').is(":checked"),
       extra_stopwords: cleanTextboxInput($('#dc-more-stops').val()),
       phrases_to_join: cleanTextboxInput($('#dc-merge').val()),
       remove_punctuation: $('#dc-punct').is(":checked"),
       do_stemming: $('#dc-stem').is(":checked"),
       do_lemmatizing: $('#dc-lemma').is(":checked"),
       min_word_length: $('#dc-min-words').val(),
    }
}


/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {
    const langSelect = document.getElementById("tm-lang");
    const langOptions = ["Arabic", "Azerbaijani", "Danish", "Dutch", "English", "Finnish", "French", "German", "Greek",
        "Hungarian", "Indonesian", "Italian", "Kazakh", "Korean", "Nepali", "Norwegian", "Portuguese", "Romanian",
        "Russian", "Slovene", "Spanish", "Swedish", "Tajik", "Turkish"];

    for (let i=0; i < langOptions.length; i++) {
        const opt = langOptions[i];
        const el = document.createElement("option");
        el.textContent = opt;
        el.value = opt;
        langSelect.appendChild(el);

    }

    $('#tm-training-visible').on('click', function () {
        $('#tm-training-invisible').click();
    });

    $("input[id='tm-training-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#tm-training-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#submit1').on('click', function () {
        // handle missing info first
        if ($('#tm-name').val() === "") {
            $('#error1-text').html('Please provide a name for your topic model.');
            $('#error1').removeClass('hidden');
        } else if ($('#tm-num').val() === "") {
            $('#error1-text').html('Please provide a number of topics to identify.');
            $('#error1').removeClass('hidden');
        } else if ($('#tm-keywords').val() === "") {
            $('#error1-text').html('Please provide a number of keywords to identify per topic.');
            $('#error1').removeClass('hidden');
        } else if (document.getElementById("tm-training-invisible").files.length === 0) {
            $('#error1-text').html('Please provide a training file.');
            $('#error1').removeClass('hidden');
        } else if ($('#tm-email').val() === "") {
            $('#error1-text').html('Please provide an email to send your results to.');
            $('#error1').removeClass('hidden');

        } else {

            $('#error1').addClass('hidden');

            // POST request for topic model
            const POST_TOPIC_MODEL = `${BASE_URL}/topic_models/`;
            let postData = {
                topic_model_name: $('#tm-name').val(),
                num_topics: $('#tm-num').val(),
                num_keywords: $('#tm-keywords').val(),
                notify_at_email: $('#tm-email').val(),
                language: $('#tm-lang').val().toLowerCase()
            };
            Object.assign(postData, getPreprocessingOptions()); // add preprocessing info

            $.ajax({
                url: POST_TOPIC_MODEL,
                type: 'POST',
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(postData),
                success: function (data) {
                    console.log('success in topic model POST');
                    // POST request for training file and preprocessing
                    const POST_TM_TRAINING_FILE = `${BASE_URL}/topic_models/${data.topic_model_id}/training/file`;
                    let fileFD = new FormData();
                    fileFD.append('file', document.getElementById("tm-training-invisible").files[0]);

                    $.ajax({
                        url: POST_TM_TRAINING_FILE,
                        data: fileFD,
                        type: 'POST',
                        processData: false,
                        contentType: false,
                        success: function(){
                            console.log('success in training file POST');
                            $('#success1').removeClass('hidden');
                        },
                        error: function (xhr, status, err) {
                            console.log(xhr.responseText);
                            let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                            $('#error1').html(`An error occurred while uploading your file: ${error}`).removeClass('hidden');
                        }
                    });
                },
                error: function (xhr, status, err) {
                    console.log(xhr.responseText);
                    let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                    $('#error1').html(`An error occurred while creating the topic model: ${error}`).removeClass('hidden');
                }
            });
        }
    });

});



