interface Friend {
  avatar: string; // 头像链接
  name: string; // 用户 id
  link: string; // 博客链接
  title?: string; // 用户头衔
  tag?: string; // 用户标签
  color?: string; // 标签颜色
}

/**
 * TODO: 缺项处理
 * 在此处填写你的友情链接
 */
export const friendsInfo: Friend[] = [
   {
     avatar: "https://avatars.githubusercontent.com/u/57825561?v=4",
     name: "Runze Cheng",
     title: "🐺 A wolf Developer",
     link: "https://aidcheng.github.io",
     tag: "UCL CS",
     color: "indigo",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/114977087?v=4",
     name: "Yuyang Song",
     title: "⚡️ LLM & Database",
     link: "https://yuyang-song.github.io",
     tag: "SCU CS",
     color: "indigo",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/141292166?v=4",
     name: "Yushen Wang",
     title: "📡 Wireless Communication",
     link: "https://sashimi-balls.github.io",
     tag: "UESTC CE",
     color: "sky",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/60817914?v=4",
     name: "Charles-Donne",
     title: "🤖 VLN & Agent & Robotics",
     link: "https://charlesdonne.vercel.app",
     tag: "CSU EE",
     color: "sky",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/195511320?v=4",
     name: "ZHANGXiyuan",
     title: "☀️ Multimodal & AI4S",
     link: "https://github.com/ZHANGXiyuan2004",
     tag: "UESTC EE",
     color: "sky",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/237438682?v=4",
     name: "Mingyang Shi",
     title: "⚙️ LLM & Model Compression",
     link: "https://mingyang2004.github.io",
     tag: "DUT SE",
     color: "indigo",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/171577079?v=4",
     name: "Yao Zhang",
     title: "💊 MLLM & Multimodal",
     link: "https://github.com/RatUestc",
     tag: "UESTC CE",
     color: "sky",
   },

];
