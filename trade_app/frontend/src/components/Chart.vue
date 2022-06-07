<template>
  <div id="app">
    <kendo-datasource
      ref="dataSource"
      :transport-read-url="'https://api.iextrading.com/1.0/stock/amzn/chart/1y'"
      :transport-read-data-type="'json'">
    </kendo-datasource>
    <kendo-stockchart
      :data-source-ref="'dataSource'"
      :title="title"
      :dateField="dateField"
      :series="series"
      :theme="'material'">
    </kendo-stockchart>
  </div>
</template>

<script>
export default {
  name: 'app',
  data () {
    return {
      title: {
        text: 'Amazon Stock Price'
      },
      dateField: 'date',
      series: [{
        type: 'candlestick',
        openField: 'open',
        highField: 'high',
        lowField: 'low',
        closeField: 'close',
        color: '#00c853',
        downColor: '#d50000'
      }]
    }
  }
}
</script>

<style>
#app {
  margin-top: 60px;
}
</style>