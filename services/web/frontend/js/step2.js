
/* * * * * */
/*  DATA   */
/* * * * * */


function getTMPreview(id) {
    console.log(id);
    let isNum = /^\d+$/.test(id);
    if (isNum === false) {
        $('#error2-text').html('Please enter a numeric value for your ID');
        $('#error2').removeClass('hidden');
    } else {
        $('#error2').addClass('hidden');
        const GET_ONE_TOPIC_MDL = BASE_URL + `/topic_models/${id}/topics/preview`;
        $.ajax({
            url: GET_ONE_TOPIC_MDL,
            type: "GET",
            dataType: "json",
            success: function (data) {
                $('#preview-info').removeClass('hidden');
                $('#tm-prev-name').empty().append(data.topic_model_name);
                $('#tm-prev-num').empty().append(data.num_topics);
                $('#previews').empty();
                $('#get-results').attr('href', `http://www.openframing.org:5000/api/topic_models/${id}/topic_zipped`);
                showPreviews(data);
            },
            error: function (xhr, status, err) {
                console.log(xhr.responseText);
                let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                $('#preview-info').addClass('hidden');
                $('#error2').html(`An error occurred while finding your topic model: ${error}`).removeClass('hidden');
            },
        });
    }
}

/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {
    const id = urlParams.get('id');

    if (id !== null) {
        // UNCOMMENT BELOW TO GET PREVIEW DATA FROM ACTUAL DB
        getTMPreview(id);

        // UNCOMMENT BELOW TO GET A PREVIEW OF WHAT THE UI WILL LOOK LIKE (NOT ACTUAL DB DATA)
        // let data = {
        //     topic_model_name: 'model1',
        //     num_topics: 2,
        //     topic_names: ['Topic 1', 'Topic 2'],
        //     topic_previews: [
        //         {
        //             keywords: ['one', 'two', 'three'],
        //             examples: ['test1', 'test2', 'test3']
        //         },
        //         {
        //             keywords: ['one', 'two', 'three'],
        //             examples: ['test1', 'test2', 'test3']
        //         },
        //     ]
        // };
        // $('#preview-info').removeClass('hidden');
        // $('#tm-prev-name').empty().append(data.topic_model_name);
        // $('#tm-prev-num').empty().append(data.num_topics);
        // $('#previews').empty();
        // showPreviews(data);


    }

    $('#submit2').on('click', function () {
        if ($('#tmp-id').val() === "") {
            $('#error2-text').html('Please enter an ID.');
            $('#error2').removeClass('hidden');

        } else {
            $('#error2').addClass('hidden');
            let id = $('#tmp-id').val();
            getTMPreview(id);
        }
    })


});


/* * * * * * * */
/*  HELPERS    */
/* * * * * * * */

function showPreviews(data) {
    let reformatted = reformatPreviewResponse(data);
    let header = formatHeader(data.topic_names.slice(0, 5)); // again taking max of 5 topics
    let body = formatBody(reformatted);

    let newTable = `
        <thead>
        ${header}
        </thead>
        <tbody>
        ${body}
        </tbody>
    `;

    $('#previews').append(newTable);
}

function formatHeader(topic_names) {
    let newRow = `<tr>\n<th scope="col">Keyword</th>\n`;
    for (let i = 0; i < topic_names.length; i++) {
        newRow += `<th scope="col">Topic ${i+1}</th>\n`
    }
    newRow += '</tr>';
    return newRow;
}

function formatBody(reformatted) {
    let newRow = ``;
    for (let i = 0; i < reformatted.length; i++) {
        newRow += `<tr>\n<th scope="row">${i+1}</th>\n`;
        for (let kw of reformatted[i]) {
            newRow += `<td>${kw}</td>`;
        }
        newRow += '</tr>\n';
    }
    return newRow;
}


function reformatPreviewResponse(data) {
    // take at most 5 topics
    let previews = data.topic_previews.slice(0, 5);
    // take at most 10 keywords
    let maxKeywords = Math.min(10, Math.max.apply(Math, previews.map(function(o) { return o.keywords.length; })));

    let reformatted = [];
    for (let i = 0; i < maxKeywords; i++) {
        let newRow = [];
        for (let pre of previews) {
            if (pre.keywords.length <= i) {
                newRow.push("");
            } else {
                newRow.push(pre.keywords[i]);
            }
        }
        reformatted.push(newRow);
    }

    return reformatted;
}






