import {createRouter, createWebHistory} from 'vue-router'
import Home from './components/Home'
import Stocks from './components/Stocks'
import StocksDetail from './components/StocksDetail'
import Signin from './components/signin.vue'
import Signup from './components/signup.vue'
import User from './components/user.vue'
import NotFound from './components/NotFound.vue'

const routes = [
    {
        path:'/',
        name:'home',
        component:Home
    },
    {
        path:'/stocks',
        name:'stocks',
        component:Stocks
    },
    {
        path:'/details/:id',
        name:'details',
        component:StocksDetail,
        props:true
    },
    { 
        path: "/signin", 
        name: "signin", 
        component: Signin 
    },
    { 
        path: "/signup", 
        name: "signup", 
        component: Signup 
    },
    { 
        path: "/user", 
        name: "user", 
        component: User 
    },
    { 
        path: "*", 
        name: "NotFound", 
        component: NotFound 
    }
]

const router = createRouter({
    history:createWebHistory(),
    routes
})

export default router;