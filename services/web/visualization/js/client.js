let BASE_URL = "http://" + window.location.host + "/api";
// uncomment below to test on AWS EC2 instance
// const BASE_URL = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api";

const urlParams = new URLSearchParams(window.location.search);

/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {
    const step = urlParams.get('step');
    const id = urlParams.get('id');

    $("#step1").load("components/step1.html", function () {
        $.getScript("js/client.js");
        $.getScript("js/step1.js");
    });

    $("#step2").load("components/step2.html", function () {
        $.getScript("js/client.js");
        $.getScript("js/step2.js");
    });

    $("#step3").load("components/step3.html");

    $("#step4").load("components/step4.html",function(){
        $.getScript("js/client.js");
        $.getScript("js/step4.js");
    });

    $("#step5").load("components/step5.html",function(){
        $.getScript("js/client.js");
        $.getScript("js/step5.js");
    });

    if (step !== null) {
        changeTabs(step);
    }


});

/* * * * * * */
/*  HELPERS  */
/* * * * * * */

function getErrorMessage(message) {
    if (typeof message === "object") {
        let strArr = [];
        for (let key of Object.keys(message)) {
            strArr.push(message[key]);
        }
        return strArr.join('; ')

    } else {
        return message;
    }
}

function cleanTextboxInput(textbox_in) {
    let values = textbox_in.split(',');
    values = values.map((cat) => {return cat.trim()});
    return values;
}

function changeTabs(step) {
    $(`#step1li`).removeClass('active');
    $(`#step1a`).removeClass('active');
    $(`#step${step}li`).addClass('active');
    $(`#step${step}a`).addClass('active');

    $(`#step1`).removeClass('active').removeClass('show');
    $(`#step${step}`).addClass('active').addClass('show');
}
