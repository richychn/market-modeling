function graph(raw) {
  var allArr = []
  for (var i = 0; i < 8; i++) {
    var str = raw[i.toString()]
    var arr = str.slice(1,-1).split(",")
    for (var j = 0; j < arr.length; j++) {
      arr[j] = parseFloat(arr[j])
    }
    allArr.push(arr)
  }
  var data = {
    series: allArr
  }
  new Chartist.Line('.ct-chart', data)
  $("#loading")[0].style.display = "none"
  $(".graph")[0].style.display = "inherit"
}

function updateall() {
  var checkboxes = document.querySelectorAll(".legend-checkbox"); //array of elements which are checkboxes
  for(var i = 0; i < checkboxes.length; i++) {
    var letter = checkboxes[i].value;
    var line = document.querySelector('.ct-series-' + letter); //gets on element called ctseriesa
    if (checkboxes[i].checked) {
      line.style.display = "inline";
    }
    else {
      line.style.display = "none";
    }
  }
}
