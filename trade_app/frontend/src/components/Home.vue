<template>
  <div class="container mt-5">
      <div v-for="user in users" :key="user.id">
        <h3>
            <router-link 
            class="link-style"
            :to="{name:'details', params:{id:user.id}}"
            >
                {{user.title}}
            </router-link>    
        </h3>
      </div>
  </div>
</template>

<script>
export default {
    data(){
        return {
            users:[]
        }
    },

    methods: {
        getContent(){
            fetch('http://localhost:5000/get', {
                method:"GET",
                headers: {
                    "Content-Type":"application/json"
                }
            })
            .then(resp => resp.json())
            .then(data => {
                this.users.push(...data)
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
.link-style {
    font-weight:bold;
    color:black;
    text-decoration: none;
}

.link-style:hover {
    color:gray;
    text-decoration: none;
}

</style>