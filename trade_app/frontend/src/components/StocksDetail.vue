<template>
  <div class="container mt-5">
      <h2>{{user.title}}</h2>
      <p class="mt-3">
          {{user.body}}
      </p>
      <h6>Date: {{user.date}}</h6>
        <h2></h2>
  </div>
</template>

<script>
export default {
    data() {
        return {
            user:{}
        }
    },
    props: {
        id: {
            type:[Number, String],
            required:true
        }
    },
    methods: {
        getStocksData() {
            fetch(`http://localhost:5000/get/${this.id}/`, {
                method:"GET",
                headers: {
                    "Content-Type":"application/json"
                }
            })
            .then(resp => resp.json())
            .then(data => {
                this.user = data
            })
            .catch(error => {
                console.log(error)
            })
        }
    },
    created() {
        this.getStocksData()
    }
}
</script>

<style>

</style>