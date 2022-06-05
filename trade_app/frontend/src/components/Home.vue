<template>
  <div class="container mt-5">
      <div v-for="user in users" :key="user.id">
        <h3>{{user.title}}</h3>
        <p>{{user.body}}</p>
        <p>{{user.date}}</p>
      </div>
  </div>
</template>

<script>
export default {
    data(){
        return {
            content:[]
        }
    },

    methods: {
        getContent(){
            fetch('http://localhost:5000/get', {
                methods:"GET",
                headers: {
                    "Content-Type":"application/json"
                }
            })
            .then(resp => resp.json())
            .then(data => {
                this.content.push(...data)
            })
            .catch(error => {
                console.log(error)
            })
        }
    },
    created(){
        this.getContent()
    }
}
</script>

<style>

</style>