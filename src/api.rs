use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "ssr")] {
        static CHARACTER_NAME: &str = "### Assistant";
        static USER_NAME: &str = "### Human";

        use std::convert::Infallible;
        use actix_web::web;
        use std::sync::Arc;
        use llm::models::Llama;
        use llm::KnownModel;
        use actix_web::HttpRequest;
        use actix_web::HttpResponse;
        use actix_web::web::Payload;
        use actix_web::Error;
        use actix_ws::Message as Msg;
        use futures::stream::{StreamExt};
        use leptos::*;

        pub fn infer(model: Arc<Llama>, inference_session: &mut llm::InferenceSession, user_message: &String, tx: tokio::sync::mpsc::Sender<String>) -> Result<(), ServerFnError> {
            use tokio::runtime::Runtime;

            // would love a way to avoid doing this if possible
            let mut runtime = Runtime::new().expect("issue creating tokio runtime");

            inference_session
                .infer(
                    model.as_ref(),
                    &mut rand::thread_rng(),
                    &llm::InferenceRequest {
                        prompt: format!("{USER_NAME}\n{user_message}\n{CHARACTER_NAME}:")
                            .as_str()
                            .into(),
                        parameters: &llm::InferenceParameters::default(),
                        play_back_previous_tokens: false,
                        maximum_token_count: None,
                    },
                    &mut Default::default(),
                    inference_callback(String::from(USER_NAME), &mut String::new(), tx, &mut runtime),
                )
                .unwrap_or_else(|e| panic!("{e}"));

            Ok(())
        }

        fn session_setup(model: Arc<Llama>) -> llm::InferenceSession {
            let persona = format!("Assistant is a large language model trained by Meta.\n\
            Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. \n\
            As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. \n\
            Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. \n\
            Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics. \n\
            Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.");

            let history = format!(
                "{CHARACTER_NAME}:Hello - How may I help you today?\n\
                {USER_NAME}:What is the capital of France?\n\
                {CHARACTER_NAME}:Paris is the capital of France.\n"
            );

            let mut session = model.start_session(Default::default());
            session
                .feed_prompt(
                    model.as_ref(),
                    format!("{persona}\n{history}").as_str(),
                    &mut Default::default(),
                    llm::feed_prompt_callback(|_| {
                        Ok::<llm::InferenceFeedback, Infallible>(llm::InferenceFeedback::Continue)
                    }),
                )
                .expect("Failed to ingest initial prompt.");

            session
        }

        fn inference_callback<'a>(
            stop_sequence: String,
            buf: &'a mut String,
            tx: tokio::sync::mpsc::Sender<String>,
            runtime: &'a mut tokio::runtime::Runtime,
        ) -> impl FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, Infallible> + 'a {
            use llm::InferenceFeedback::Halt;
            use llm::InferenceFeedback::Continue;

            move |resp| -> Result<llm::InferenceFeedback, Infallible> {match resp {
                llm::InferenceResponse::InferredToken(t) => {
                    let mut reverse_buf = buf.clone();
                    reverse_buf.push_str(t.as_str());
                    if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                        buf.clear();
                        return Ok(Halt);
                    } else if stop_sequence.as_str().starts_with(reverse_buf.as_str()) {
                        buf.push_str(t.as_str());
                        return Ok(Continue);
                    }

                    // Clone the string we're going to send
                    let text_to_send = if buf.is_empty() {
                        t.clone()
                    } else {
                        reverse_buf
                    };

                    let tx_cloned = tx.clone();
                    runtime.block_on(async move {
                        tx_cloned.send(text_to_send).await.expect("issue sending on channel");
                    });

                    Ok(Continue)
                }
                llm::InferenceResponse::EotToken => Ok(Halt),
                _ => Ok(Continue),
            }}
        }

        pub async fn ws(req: HttpRequest, body: Payload, model: web::Data<Llama>) -> Result<HttpResponse, Error> {
            let (response, session, mut msg_stream) = actix_ws::handle(&req, body)?;

            use std::sync::Mutex;
            use tokio::sync::mpsc;

            let (send_inference, mut recieve_inference) = mpsc::channel(100);

            let mdl: Arc<Llama> = model.into_inner().clone();
            let sess = Arc::new(Mutex::new(session));
            let sess_cloned = sess.clone();
            actix_rt::spawn(async move {
                let (send_new_user_message, recieve_new_user_message) =
                    std::sync::mpsc::channel();

                // Rustformers sessions need to stay on the same thread
                // So we can't really rely on TOKIO
                let model_cloned = mdl.clone();
                // let send_inference_cloned = send_inference.clone();

                std::thread::spawn(move || {
                    let mut inference_session = session_setup(mdl);

                    for new_user_message in recieve_new_user_message {
                        let _ = infer(model_cloned.clone(), &mut inference_session, &new_user_message, send_inference.clone());
                    }
                });

                while let Some(Ok(msg)) = msg_stream.next().await {
                    match msg {
                        Msg::Ping(bytes) => {
                            let res = sess_cloned.lock().unwrap().pong(&bytes).await;
                            if res.is_err() {
                                return;
                            }
                        }
                        Msg::Text(s) => {
                            // send it to the dedicated inference thread
                            let _ = send_new_user_message.send(s.to_string());
                        }
                        _ => break,
                    }
                }
            });

            // another task to receive inferred tokens and send them
            // over the WebSocket connection while the inference
            // itself chugs away on a separate thread
            actix_rt::spawn(async move {
                while let Some(message) = recieve_inference.recv().await {
                    sess.lock().unwrap().text(message).await.expect("issue sending on websocket");
                }
                // let _ = sess.lock().unwrap().close(None).await;
            });

            Ok(response)
        }
    }
}
