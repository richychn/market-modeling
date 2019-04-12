function graph(raw) {
  var data = {
    series: [raw]
  }
  new Chartist.Line('.ct-chart', data)
}
