
/* * * * * */
/*  DATA   */
/* * * * * */


function submitTestSet(id, spinnerId, errorId, submitId) {
    let isNum = /^\d+$/.test(id);
    if (isNum === false) {
        $('#error5-text').html('Please enter a numeric value for your ID');
        $('#error5').removeClass('hidden');
    } else {
        $('#error5').addClass('hidden');

        // POST request for topic model
        const POST_TEST_SET = `${BASE_URL}/classifiers/${id}/test_sets/`;
        let postData = {
            test_set_name: $('#pt-name').val(),
            notify_at_email: $('#pt-email').val()
        };
        $.ajax({
            url: POST_TEST_SET,
            type: 'POST',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify(postData),
            success: function (data) {
                console.log('success in classifier test set POST');
                // POST request for training file
                let c_id = data.classifier_id;
                let t_id = data.test_set_id;
                const POST_PT_TESTING_FILE = `${BASE_URL}/classifiers/${c_id}/test_sets/${t_id}/file`;
                let fileFD = new FormData();
                fileFD.append('file', document.getElementById("pt-testing-invisible").files[0]);

                $.ajax({
                    url: POST_PT_TESTING_FILE,
                    data: fileFD,
                    type: 'POST',
                    processData: false,
                    contentType: false,
                    success: function () {
                        console.log('STEP 5 - success in testing file POST');
                        pingClassifierStatus(c_id, t_id);
                        $(spinnerId).hide();
                        $(submitId).removeClass("disabled");
                    },
                    error: function (xhr, status, err) {
                        console.log(xhr.responseText);
                        let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                        $(errorId).html(`An error occurred while uploading your file: ${error}`).removeClass('hidden');
                        $(spinnerId).hide();
                        $(submitId).removeClass("disabled");
                    }
                });
            },
            error: function (xhr, status, err) {
                console.log(xhr.responseText);
                let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                $(errorId).html(`An error occurred while creating the test set: ${error}`).removeClass('hidden');
                $(spinnerId).hide();
                $(submitId).removeClass("disabled");
            }
        });
    }
}


function pingClassifierStatus(c_id, t_id) {
    let stop = false;
    let interval = setInterval(function () {
        const GET_TEST_SET = `${BASE_URL}/classifiers/${c_id}/test_sets/${t_id}`;
        if (stop) {
            clearInterval(interval);
        } else {
            $.ajax({
                url: GET_TEST_SET,
                type: 'GET',
                success: function (data) {
                    if (data.status === "completed") {
                        stop = true;
                        window.location.replace(`${BASE_URL}/classifiers/${c_id}/test_sets/${t_id}/predictions`);
                    }
                },
                error: function (xhr, status, err) {
                    console.log(xhr.responseText);
                    let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                    $('#error5').html(`An error occurred while checking the status of your test set: ${error}`).removeClass('hidden');
                }
            });
        }
    }, 2000);
}


/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {
    const id = urlParams.get('id');


    $('#pt-testing-visible').on('click', function () {
        $('#pt-testing-invisible').click();
    });

    $("input[id='pt-testing-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#pt-testing-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#pt-testing-visible-2').on('click', function () {
        $('#pt-testing-invisible-2').click();
    });

    $("input[id='pt-testing-invisible-2']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#pt-testing-filepath-2')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });


    $('#submit5').on('click', function () {
        // handle missing info first
        if ($('#pt-id').val() === "") {
            $('#error5-text').html('Please enter an ID.');
            $('#error5').removeClass('hidden');
        } else if (document.getElementById("pt-testing-invisible").files.length === 0) {
            $('#error5-text').html('Please provide a test file.');
            $('#error5').removeClass('hidden');
        } else if ($('#pt-name').val() === "") {
            $('#error5-text').html('Please name your test set.');
            $('#error5').removeClass('hidden');
        } else if ($('#pt-email').val() === "") {
            $('#error5-text').html('Please provide an email address.');
            $('#error5').removeClass('hidden');
        } else {

            $('#error5').addClass('hidden');
            $('#pt-spinner').show();
            $('#submit5').addClass("disabled");

            const id = $('#pt-id').val();
            submitTestSet(id, '#pt-spinner', '#error5', 'submit5');

        }
    });

    $('#submit5-2').on('click', function () {
        // console.log($("input[type='radio'][name='policyissue']:checked").val());
        // handle missing info first
        if ($("input[type='radio'][name='policyissue']:checked").val() === undefined) {
            $('#error5-text-2').html('Please select a pretrained classifier.');
            $('#error5-2').removeClass('hidden');
        } else if (document.getElementById("pt-testing-invisible-2").files.length === 0) {
            $('#error5-text-2').html('Please provide a test file.');
            $('#error5-2').removeClass('hidden');
        } else if ($('#pt-name-2').val() === "") {
            $('#error5-text-2').html('Please name your test set.');
            $('#error5-2').removeClass('hidden');
        } else if ($('#pt-email-2').val() === "") {
            $('#error5-text-2').html('Please provide an email address.');
            $('#error5-2').removeClass('hidden');
        } else {

            $('#error5-2').addClass('hidden');
            $('#pt-spinner-2').show();
            $('#submit5-2').addClass("disabled");

            const id = $("input[type='radio'][name='policyissue']:checked").val();
            submitTestSet(id, '#pt-spinner-2', '#error5-2', 'submit5-2');

        }
    })

});


/* * * * * * * */
/*  HELPERS    */
/* * * * * * * */


function clearOptions() {
    $("input[name='policyissue']:checked").val([]);
}

